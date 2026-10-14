import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from socceraction.vaep import VAEP
import socceraction.vaep.formula as vaepformula
from socceraction.atomic.vaep import AtomicVAEP
import socceraction.atomic.vaep.formula as atomic_vaepformula

warnings.filterwarnings("ignore")

DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def extract_features_and_labels(df_games: pd.DataFrame, df_actions: pd.DataFrame, engine_type: str = "standard"):
    """
    逐場計算遊戲狀態特徵 (Game State Features) 與未來 10 動作進球/失球標籤 (Labels)。
    engine_type: 'standard' (VAEP) 或 'atomic' (Atomic-VAEP)
    """
    engine = VAEP(nb_prev_actions=3) if engine_type == "standard" else AtomicVAEP(nb_prev_actions=3)
    x_list, y_list, meta_list = [], [], []

    grouped_actions = {gid: grp.reset_index(drop=True) for gid, grp in df_actions.groupby("game_id", sort=False)}

    for _, game in tqdm(df_games.iterrows(), total=len(df_games), desc=f"Extracting {engine_type.upper()} features"):
        gid = game["game_id"]
        if gid not in grouped_actions:
            continue
        g_actions = grouped_actions[gid]

        X_g = engine.compute_features(game, g_actions)
        Y_g = engine.compute_labels(game, g_actions)

        x_list.append(X_g)
        y_list.append(Y_g)
        meta_list.append(g_actions)

    X_all = pd.concat(x_list, ignore_index=True)

    # 將 AtomicVAEP 中的 Categorical / 字串欄位 (如 'pass', 'foot') 自動展開為 One-Hot 數值特徵
    cat_cols = X_all.select_dtypes(include=["category", "object"]).columns.tolist()
    if len(cat_cols) > 0:
        X_all = pd.get_dummies(X_all, columns=cat_cols, dtype=np.float32)

    # 轉換布林與數值特徵為 float32，並清理極端值確保 XGBoost / CatBoost 相容性
    X_all = X_all.astype(np.float32).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    Y_all = pd.concat(y_list, ignore_index=True)
    Y_all["scores"] = Y_all["scores"].astype(int)
    Y_all["concedes"] = Y_all["concedes"].astype(int)

    df_aligned_actions = pd.concat(meta_list, ignore_index=True)
    return X_all, Y_all, df_aligned_actions


def train_and_evaluate_models(X: pd.DataFrame, Y: pd.DataFrame, df_actions: pd.DataFrame, train_gids: set, framework_name: str):
    """
    在比賽層級切分之訓練集上訓練 XGBoost 與 CatBoost，並在測試集評估 ROC-AUC、Brier Score 與 Log-Loss。
    回傳全資料集之預測機率與評估指標清單。
    """
    is_train = df_actions["game_id"].isin(train_gids).values
    X_train, X_test = X[is_train], X[~is_train]
    Y_train, Y_test = Y[is_train], Y[~is_train]

    learners = {
        "XGBoost": lambda: XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
        "CatBoost": lambda: CatBoostClassifier(
            iterations=250,
            depth=6,
            learning_rate=0.1,
            loss_function="Logloss",
            verbose=0,
            random_seed=42,
            thread_count=-1,
        ),
    }

    preds_all = {}
    metrics_records = []
    feature_importances = {}

    for learner_name, model_fn in learners.items():
        preds_all[learner_name] = {}
        for target in ["scores", "concedes"]:
            print(f"[INFO] Training {framework_name} | {learner_name} | Target: {target}...")
            clf = model_fn()
            clf.fit(X_train, Y_train[target])

            # 測試集預測機率與指標評估
            prob_test = clf.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(Y_test[target], prob_test)
            brier = brier_score_loss(Y_test[target], prob_test)
            lloss = log_loss(Y_test[target], prob_test)

            metrics_records.append({
                "Framework": framework_name,
                "Learner": learner_name,
                "Target": target,
                "Test_ROC_AUC": auc,
                "Test_Brier_Score": brier,
                "Test_Log_Loss": lloss,
                "Positive_Rate_Pct": Y_test[target].mean() * 100.0,
            })

            # 全資料集推論以計算所有比賽的動作價值
            preds_all[learner_name][target] = clf.predict_proba(X)[:, 1]

            if target == "scores" and learner_name == "XGBoost":
                feature_importances[framework_name] = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)

    return preds_all, metrics_records, feature_importances


def compute_vaep_per_game(df_actions: pd.DataFrame, p_scores: np.ndarray, p_concedes: np.ndarray, is_atomic: bool = False):
    """
    逐場呼叫 formula.value()，避免跨比賽邊界相減產生誤差。
    """
    formula_mod = atomic_vaepformula if is_atomic else vaepformula
    df_temp = df_actions.copy()
    df_temp["_ps"] = p_scores
    df_temp["_pc"] = p_concedes

    val_list = []
    for _, grp in df_temp.groupby("game_id", sort=False):
        grp_clean = grp.reset_index(drop=True)
        vals = formula_mod.value(grp_clean, grp_clean["_ps"], grp_clean["_pc"])
        vals.index = grp.index
        val_list.append(vals)

    return pd.concat(val_list).sort_index()


def run_vaep_pipeline():
    print("[INFO] Loading processed datasets from Stage 1 & Stage 2...")
    df_games = pd.read_parquet(os.path.join(DATA_DIR, "games.parquet"))
    df_players = pd.read_parquet(os.path.join(DATA_DIR, "players.parquet"))
    df_teams = pd.read_parquet(os.path.join(DATA_DIR, "teams.parquet"))
    df_spadl_xt = pd.read_parquet(os.path.join(DATA_DIR, "spadl_actions_with_xt.parquet"))
    df_spadl_raw = pd.read_parquet(os.path.join(DATA_DIR, "spadl_actions.parquet"))
    df_atomic_raw = pd.read_parquet(os.path.join(DATA_DIR, "atomic_spadl_actions.parquet"))

    # 以比賽為單位切分 80% 訓練集 (132 場) 與 20% 測試集 (34 場)
    train_games, test_games = train_test_split(
        df_games, test_size=0.20, random_state=42, stratify=df_games["tournament_name"]
    )
    train_gids = set(train_games["game_id"].unique())
    print(f"[INFO] Match-level split: {len(train_games)} Train matches | {len(test_games)} Test matches")

    # 1. 擷取 Standard VAEP 特徵與標籤
    X_std, Y_std, df_spadl_aligned = extract_features_and_labels(df_games, df_spadl_raw, engine_type="standard")
    # 將 Stage 2 算出的連續與離散 xT 對應合併過來
    df_spadl_aligned["xt_discrete"] = df_spadl_xt["xt_discrete"].values
    df_spadl_aligned["xt_continuous"] = df_spadl_xt["xt_continuous"].values

    # 2. 訓練與評估 Standard VAEP (XGBoost & CatBoost)
    preds_std, metrics_std, feat_imp_std = train_and_evaluate_models(
        X_std, Y_std, df_spadl_aligned, train_gids, framework_name="Standard-VAEP"
    )

    # 計算每個 SPADL 動作的 VAEP 價值
    for learner_Key, suffix in [("XGBoost", "xgb"), ("CatBoost", "cat")]:
        vals = compute_vaep_per_game(
            df_spadl_aligned,
            preds_std[learner_Key]["scores"],
            preds_std[learner_Key]["concedes"],
            is_atomic=False,
        )
        df_spadl_aligned[f"p_scores_{suffix}"] = preds_std[learner_Key]["scores"]
        df_spadl_aligned[f"p_concedes_{suffix}"] = preds_std[learner_Key]["concedes"]
        df_spadl_aligned[f"offensive_value_{suffix}"] = vals["offensive_value"]
        df_spadl_aligned[f"defensive_value_{suffix}"] = vals["defensive_value"]
        df_spadl_aligned[f"vaep_value_{suffix}"] = vals["vaep_value"]

    # 3. 擷取 Atomic-VAEP 特徵與標籤
    X_atm, Y_atm, df_atomic_aligned = extract_features_and_labels(df_games, df_atomic_raw, engine_type="atomic")

    # 4. 訓練與評估 Atomic-VAEP (XGBoost & CatBoost)
    preds_atm, metrics_atm, feat_imp_atm = train_and_evaluate_models(
        X_atm, Y_atm, df_atomic_aligned, train_gids, framework_name="Atomic-VAEP"
    )

    # 計算每個 Atomic-SPADL 動作的價值
    for learner_Key, suffix in [("XGBoost", "xgb"), ("CatBoost", "cat")]:
        vals_atm = compute_vaep_per_game(
            df_atomic_aligned,
            preds_atm[learner_Key]["scores"],
            preds_atm[learner_Key]["concedes"],
            is_atomic=True,
        )
        df_atomic_aligned[f"p_scores_{suffix}"] = preds_atm[learner_Key]["scores"]
        df_atomic_aligned[f"p_concedes_{suffix}"] = preds_atm[learner_Key]["concedes"]
        df_atomic_aligned[f"offensive_value_{suffix}"] = vals_atm["offensive_value"]
        df_atomic_aligned[f"defensive_value_{suffix}"] = vals_atm["defensive_value"]
        df_atomic_aligned[f"vaep_value_{suffix}"] = vals_atm["vaep_value"]

    # 5. 儲存評估指標與動作估值資料表
    df_metrics = pd.DataFrame(metrics_std + metrics_atm)
    df_metrics.to_csv(os.path.join(MODEL_DIR, "vaep_model_benchmarks.csv"), index=False)

    # 將動作方向轉為標準 Left-to-Right 後儲存供 Streamlit 視覺化使用
    import socceraction.spadl as spadl
    import socceraction.atomic.spadl as atomicspadl

    ltr_spadl = []
    for gid, grp in df_spadl_aligned.groupby("game_id", sort=False):
        htid = grp["home_team_id"].iloc[0]
        g_ltr = spadl.play_left_to_right(grp, home_team_id=htid)
        ltr_spadl.append(g_ltr)
    df_spadl_final = pd.concat(ltr_spadl, ignore_index=True)

    ltr_atomic = []
    for gid, grp in df_atomic_aligned.groupby("game_id", sort=False):
        htid = grp["home_team_id"].iloc[0]
        g_ltr = atomicspadl.play_left_to_right(grp, home_team_id=htid)
        ltr_atomic.append(g_ltr)
    df_atomic_final = pd.concat(ltr_atomic, ignore_index=True)

    df_spadl_final.to_parquet(os.path.join(DATA_DIR, "spadl_actions_valued.parquet"), index=False)
    df_atomic_final.to_parquet(os.path.join(DATA_DIR, "atomic_actions_valued.parquet"), index=False)

    # 6. 彙整球員全維度評價總表 (Player Summary Ratings)
    player_meta = (
        df_players.groupby("player_id")
        .agg(
            player_name=("player_name", "first"),
            total_minutes=("minutes_played", "sum"),
            matches_played=("game_id", "nunique"),
        )
        .reset_index()
    )

    # 找出每位球員最常代表的國家隊
    player_team = (
        df_players.merge(df_teams, on="team_id", how="left")
        .groupby("player_id")["team_name"]
        .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Unknown")
        .reset_index()
    )

    std_player = (
        df_spadl_final.groupby("player_id")
        .agg(
            spadl_actions=("action_id", "count"),
            xt_discrete_sum=("xt_discrete", "sum"),
            xt_cont_sum=("xt_continuous", "sum"),
            vaep_xgb_sum=("vaep_value_xgb", "sum"),
            off_xgb_sum=("offensive_value_xgb", "sum"),
            def_xgb_sum=("defensive_value_xgb", "sum"),
            vaep_cat_sum=("vaep_value_cat", "sum"),
            off_cat_sum=("offensive_value_cat", "sum"),
            def_cat_sum=("defensive_value_cat", "sum"),
        )
        .reset_index()
    )

    atm_player = (
        df_atomic_final.groupby("player_id")
        .agg(
            atomic_actions=("action_id", "count"),
            atomic_vaep_xgb_sum=("vaep_value_xgb", "sum"),
            atomic_off_xgb_sum=("offensive_value_xgb", "sum"),
            atomic_def_xgb_sum=("defensive_value_xgb", "sum"),
            atomic_vaep_cat_sum=("vaep_value_cat", "sum"),
            atomic_off_cat_sum=("offensive_value_cat", "sum"),
            atomic_def_cat_sum=("defensive_value_cat", "sum"),
        )
        .reset_index()
    )

    df_summary = (
        player_meta.merge(player_team, on="player_id", how="left")
        .merge(std_player, on="player_id", how="left")
        .merge(atm_player, on="player_id", how="left")
        .fillna(0.0)
    )

    mins_safe = df_summary["total_minutes"].replace(0, np.nan)
    df_summary["xt_cont_p90"] = (df_summary["xt_cont_sum"] / mins_safe) * 90.0
    df_summary["vaep_xgb_p90"] = (df_summary["vaep_xgb_sum"] / mins_safe) * 90.0
    df_summary["vaep_cat_p90"] = (df_summary["vaep_cat_sum"] / mins_safe) * 90.0
    df_summary["off_xgb_p90"] = (df_summary["off_xgb_sum"] / mins_safe) * 90.0
    df_summary["def_xgb_p90"] = (df_summary["def_xgb_sum"] / mins_safe) * 90.0
    df_summary["atomic_vaep_xgb_p90"] = (df_summary["atomic_vaep_xgb_sum"] / mins_safe) * 90.0
    df_summary["atomic_vaep_cat_p90"] = (df_summary["atomic_vaep_cat_sum"] / mins_safe) * 90.0

    df_summary.to_parquet(os.path.join(DATA_DIR, "player_summary_ratings.parquet"), index=False)

    # 7. 繪製高解析度學術圖表 Figure 2: Multi-Model Valuation Benchmark
    print("[INFO] Generating publication Figure 2: xT vs. VAEP vs. Atomic-VAEP Benchmark...")
    qual = df_summary[df_summary["total_minutes"] >= 450].copy()

    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=300)

    # Panel A: ROC-AUC Benchmark Bar Chart
    ax0 = axes[0, 0]
    labels_bar = [f"{r['Framework']}\n({r['Learner']})" for _, r in df_metrics[df_metrics["Target"] == "scores"].iterrows()]
    scores_auc = df_metrics[df_metrics["Target"] == "scores"]["Test_ROC_AUC"].values
    concedes_auc = df_metrics[df_metrics["Target"] == "concedes"]["Test_ROC_AUC"].values
    x_pos = np.arange(len(labels_bar))
    w = 0.35
    ax0.bar(x_pos - w / 2, scores_auc, w, label="P(Scores) ROC-AUC", color="#2563EB")
    ax0.bar(x_pos + w / 2, concedes_auc, w, label="P(Concedes) ROC-AUC", color="#0D9488")
    ax0.set_xticks(x_pos)
    ax0.set_xticklabels(labels_bar, fontsize=9.5, weight="bold")
    ax0.set_ylim(0.70, 0.90)
    ax0.set_ylabel("Out-of-Sample ROC-AUC", fontsize=10, weight="bold")
    ax0.set_title("A. Out-of-Sample Classification ROC-AUC (34 Unseen Matches)", fontsize=11.5, weight="bold")
    ax0.grid(axis="y", linestyle="--", alpha=0.5)
    ax0.legend(loc="lower right")

    # Panel B: Continuous xT per 90 vs. Standard VAEP per 90
    ax1 = axes[0, 1]
    ax1.scatter(qual["xt_cont_p90"], qual["vaep_xgb_p90"], alpha=0.55, color="#4F46E5", edgecolors="none", s=35)
    top_highlight = qual.nlargest(8, "vaep_xgb_p90")
    for _, row in top_highlight.iterrows():
        short_n = row["player_name"].split()[-1]
        ax1.annotate(short_n, (row["xt_cont_p90"], row["vaep_xgb_p90"]), fontsize=8, weight="bold", xytext=(4, 4), textcoords="offset points")
    ax1.set_xlabel("Continuous xT per 90 Mins (Progression Only)", fontsize=10, weight="bold")
    ax1.set_ylabel("VAEP per 90 Mins (All On-Ball Actions)", fontsize=10, weight="bold")
    ax1.set_title("B. Continuous xT vs. Standard VAEP (Min. 450 Mins)", fontsize=11.5, weight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Panel C: Offensive vs. Defensive VAEP per 90
    ax2 = axes[1, 0]
    ax2.scatter(qual["off_xgb_p90"], qual["def_xgb_p90"], alpha=0.55, color="#D97706", edgecolors="none", s=35)
    for _, row in pd.concat([qual.nlargest(5, "off_xgb_p90"), qual.nlargest(5, "def_xgb_p90")]).drop_duplicates("player_id").iterrows():
        short_n = row["player_name"].split()[-1]
        ax2.annotate(short_n, (row["off_xgb_p90"], row["def_xgb_p90"]), fontsize=8, weight="bold", xytext=(4, 4), textcoords="offset points")
    ax2.set_xlabel("Offensive VAEP per 90 Mins (ΔP_score)", fontsize=10, weight="bold")
    ax2.set_ylabel("Defensive VAEP per 90 Mins (-ΔP_concede)", fontsize=10, weight="bold")
    ax2.set_title("C. Offensive vs. Defensive Action Valuation Decomposition", fontsize=11.5, weight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Panel D: Standard VAEP vs. Atomic-VAEP per 90
    ax3 = axes[1, 1]
    ax3.scatter(qual["vaep_xgb_p90"], qual["atomic_vaep_xgb_p90"], alpha=0.55, color="#DC2626", edgecolors="none", s=35)
    for _, row in qual.nlargest(7, "atomic_vaep_xgb_p90").iterrows():
        short_n = row["player_name"].split()[-1]
        ax3.annotate(short_n, (row["vaep_xgb_p90"], row["atomic_vaep_xgb_p90"]), fontsize=8, weight="bold", xytext=(4, 4), textcoords="offset points")
    ax3.set_xlabel("Standard VAEP per 90 Mins (XGBoost)", fontsize=10, weight="bold")
    ax3.set_ylabel("Atomic-VAEP per 90 Mins (XGBoost)", fontsize=10, weight="bold")
    ax3.set_title("D. Standard VAEP vs. Atomic-VAEP Player Valuation Alignment", fontsize=11.5, weight="bold")
    ax3.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Multi-Model Action Valuation Benchmark: Continuous xT, Standard VAEP, and Atomic-VAEP", fontsize=14, weight="bold", y=0.99)
    plt.tight_layout()
    fig2_path = os.path.join(MODEL_DIR, "fig2_vaep_xt_benchmark.png")
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 8. 印出完整實驗結果供技術報告使用
    r_xt_vaep = np.corrcoef(qual["xt_cont_p90"], qual["vaep_xgb_p90"])[0, 1]
    r_vaep_atomic = np.corrcoef(qual["vaep_xgb_p90"], qual["atomic_vaep_xgb_p90"])[0, 1]
    r_xgb_cat = np.corrcoef(qual["vaep_xgb_p90"], qual["vaep_cat_p90"])[0, 1]

    print("\n" + "=" * 80)
    print("STAGE 3: VAEP & ATOMIC-VAEP PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Standard VAEP Feature Matrix Shape    : {X_std.shape}")
    print(f"Atomic-VAEP Feature Matrix Shape      : {X_atm.shape}")
    print(f"Player Rating Correlations (Min 450m) : xT vs VAEP r={r_xt_vaep:.4f} | VAEP vs Atomic r={r_vaep_atomic:.4f} | XGB vs Cat r={r_xgb_cat:.4f}")
    print("-" * 80)
    print("OUT-OF-SAMPLE MODEL BENCHMARK (34 TEST MATCHES):")
    print(df_metrics.to_string(index=False))
    print("-" * 80)
    print("TOP 10 PLAYERS BY TOTAL STANDARD VAEP (XGBoost):")
    top_vaep = df_summary.sort_values(by="vaep_xgb_sum", ascending=False).head(10)
    print(top_vaep[["player_name", "team_name", "total_minutes", "off_xgb_sum", "def_xgb_sum", "vaep_xgb_sum", "vaep_xgb_p90"]].to_string(index=False))
    print("-" * 80)
    print("TOP 10 PLAYERS BY TOTAL ATOMIC-VAEP (XGBoost):")
    top_atomic = df_summary.sort_values(by="atomic_vaep_xgb_sum", ascending=False).head(10)
    print(top_atomic[["player_name", "team_name", "total_minutes", "atomic_off_xgb_sum", "atomic_def_xgb_sum", "atomic_vaep_xgb_sum", "atomic_vaep_xgb_p90"]].to_string(index=False))
    print("-" * 80)
    print("TOP 10 DEFENSIVE CONTRIBUTORS BY DEFENSIVE VAEP (Min. 450 mins):")
    top_def = qual.sort_values(by="def_xgb_sum", ascending=False).head(10)
    print(top_def[["player_name", "team_name", "total_minutes", "def_xgb_sum", "def_xgb_p90", "vaep_xgb_sum"]].to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    run_vaep_pipeline()