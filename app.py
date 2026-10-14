import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from scipy.interpolate import RegularGridInterpolator

st.set_page_config(
    page_title="Spatiotemporal Soccer Analytics | xT & (Atomic-)VAEP",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 自動鎖定 app.py 所在的專案根目錄，避免終端機工作目錄不同導致 FileNotFoundError
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# 若先前在使用者家目錄下執行過腳本，自動支援備援路徑讀取
if not os.path.exists(os.path.join(DATA_DIR, "games.parquet")) and os.path.exists(os.path.expanduser("~/data/games.parquet")):
    DATA_DIR = os.path.expanduser("~/data")
    MODEL_DIR = os.path.expanduser("~/models")

FIELD_LENGTH = 105.0
FIELD_WIDTH = 68.0


@st.cache_data
def load_core_data():
    df_games = pd.read_parquet(os.path.join(DATA_DIR, "games.parquet"))
    df_teams = pd.read_parquet(os.path.join(DATA_DIR, "teams.parquet"))
    df_players = pd.read_parquet(os.path.join(DATA_DIR, "players.parquet"))
    df_summary = pd.read_parquet(os.path.join(DATA_DIR, "player_summary_ratings.parquet"))
    df_metrics = pd.read_csv(os.path.join(MODEL_DIR, "vaep_model_benchmarks.csv"))
    xt_disc = np.load(os.path.join(MODEL_DIR, "xt_discrete_matrix.npy"))
    xt_smooth = np.load(os.path.join(MODEL_DIR, "xt_smoothed_matrix.npy"))

    team_map = dict(zip(df_teams["team_id"], df_teams["team_name"]))
    df_games["home_name"] = df_games["home_team_id"].map(team_map)
    df_games["away_name"] = df_games["away_team_id"].map(team_map)
    df_games["match_label"] = (
        df_games["tournament_name"]
        + " | "
        + df_games["home_name"]
        + " vs. "
        + df_games["away_name"]
        + " ("
        + df_games["home_score"].astype(str)
        + "-"
        + df_games["away_score"].astype(str)
        + ") [ID:"
        + df_games["game_id"].astype(str)
        + "]"
    )
    return df_games, df_teams, df_players, df_summary, df_metrics, xt_disc, xt_smooth


@st.cache_data
def load_match_actions(game_id: int):
    df_spadl = pd.read_parquet(
        os.path.join(DATA_DIR, "spadl_actions_valued.parquet"),
        filters=[("game_id", "==", game_id)],
    )
    df_players = pd.read_parquet(os.path.join(DATA_DIR, "players.parquet"))
    df_teams = pd.read_parquet(os.path.join(DATA_DIR, "teams.parquet"))
    p_map = df_players.groupby("player_id")["player_name"].first().to_dict()
    t_map = dict(zip(df_teams["team_id"], df_teams["team_name"]))

    df_spadl["player_name"] = df_spadl["player_id"].map(p_map).fillna("Unknown")
    df_spadl["team_name"] = df_spadl["team_id"].map(t_map).fillna("Unknown")
    period_offsets = {1: 0.0, 2: 45.0, 3: 90.0, 4: 105.0, 5: 120.0}
    df_spadl["match_minute"] = (
        df_spadl["period_id"].map(period_offsets).fillna(0.0) + df_spadl["time_seconds"] / 60.0
    )
    return df_spadl.sort_values("action_id").reset_index(drop=True)


def get_interpolator(xt_smooth: np.ndarray):
    n_y, n_x = xt_smooth.shape
    y_centers = np.linspace(FIELD_WIDTH / (2 * n_y), FIELD_WIDTH - FIELD_WIDTH / (2 * n_y), n_y)
    x_centers = np.linspace(FIELD_LENGTH / (2 * n_x), FIELD_LENGTH - FIELD_LENGTH / (2 * n_x), n_x)
    return RegularGridInterpolator((y_centers, x_centers), xt_smooth, method="linear", bounds_error=False, fill_value=None)


def main():
    df_games, df_teams, df_players, df_summary, df_metrics, xt_disc, xt_smooth = load_core_data()
    cont_interp = get_interpolator(xt_smooth)

    st.title("Spatiotemporal Soccer Analytics Engine: Continuous xT & (Atomic-)VAEP")
    st.markdown(
        "**Empirical Dataset:** 166 Matches (FIFA World Cup 2022, UEFA Euro 2024, UEFA Euro 2020) | "
        "**615,225 Raw Events** | **361,378 SPADL Actions** | **542,800 Atomic-SPADL Actions**"
    )

    tab1, tab2, tab3 = st.tabs([
        "1. Multi-Model Player Benchmarking",
        "2. Match Trajectory & Action-Value Timeline",
        "3. Interactive Pass-Threat (ΔxT) Calculator",
    ])

    # ---------------- TAB 1: PLAYER BENCHMARKING ----------------
    with tab1:
        col_ctrl, col_viz = st.columns([1, 2.3])
        with col_ctrl:
            st.subheader("Filter & Model Controls")
            min_mins = st.slider("Minimum Minutes Played", min_value=90, max_value=1200, value=450, step=90)
            metric_options = {
                "Continuous xT (Cumulative)": "xt_cont_sum",
                "Continuous xT per 90": "xt_cont_p90",
                "Standard VAEP [XGBoost] (Cumulative)": "vaep_xgb_sum",
                "Standard VAEP per 90 [XGBoost]": "vaep_xgb_p90",
                "Standard VAEP per 90 [CatBoost]": "vaep_cat_p90",
                "Offensive VAEP per 90 [XGBoost]": "off_xgb_p90",
                "Defensive VAEP per 90 [XGBoost]": "def_xgb_p90",
                "Atomic-VAEP [XGBoost] (Cumulative)": "atomic_vaep_xgb_sum",
                "Atomic-VAEP per 90 [XGBoost]": "atomic_vaep_xgb_p90",
                "Atomic-VAEP per 90 [CatBoost]": "atomic_vaep_cat_p90",
            }
            x_label = st.selectbox("X-Axis Metric", list(metric_options.keys()), index=1)
            y_label = st.selectbox("Y-Axis Metric", list(metric_options.keys()), index=3)
            rank_label = st.selectbox("Leaderboard Sorting Metric", list(metric_options.keys()), index=8)

            st.markdown("#### Out-of-Sample Model Benchmarks (34 Test Matches)")
            st.dataframe(
                df_metrics[["Framework", "Learner", "Target", "Test_ROC_AUC", "Test_Brier_Score"]].style.format(
                    {"Test_ROC_AUC": "{:.4f}", "Test_Brier_Score": "{:.5f}"}
                ),
                width="stretch",
                hide_index=True,
            )

        with col_viz:
            df_filt = df_summary[df_summary["total_minutes"] >= min_mins].copy()
            x_col, y_col, r_col = metric_options[x_label], metric_options[y_label], metric_options[rank_label]

            fig, ax = plt.subplots(figsize=(9, 4.8), dpi=180)
            ax.scatter(df_filt[x_col], df_filt[y_col], alpha=0.65, color="#2563EB", s=45, edgecolors="white", linewidth=0.5)
            top_annot = df_filt.nlargest(12, r_col)
            for _, r in top_annot.iterrows():
                ax.annotate(
                    r["player_name"].split()[-1],
                    (r[x_col], r[y_col]),
                    fontsize=8,
                    weight="bold",
                    xytext=(4, 4),
                    textcoords="offset points",
                )
            ax.set_xlabel(x_label, fontsize=10, weight="bold")
            ax.set_ylabel(y_label, fontsize=10, weight="bold")
            ax.set_title(f"Player Valuation Comparison ({len(df_filt)} Players >= {min_mins} mins)", fontsize=11, weight="bold")
            ax.grid(True, linestyle="--", alpha=0.5)
            st.pyplot(fig)

        st.subheader(f"Top 20 Players Ranked by: {rank_label}")
        disp_cols = [
            "player_name", "team_name", "total_minutes", "matches_played",
            "xt_cont_sum", "xt_cont_p90", "vaep_xgb_sum", "vaep_xgb_p90",
            "def_xgb_sum", "atomic_vaep_xgb_sum", "atomic_vaep_xgb_p90"
        ]
        st.dataframe(
            df_filt.sort_values(by=r_col, ascending=False)[disp_cols].head(20).reset_index(drop=True),
            width="stretch",
        )

    # ---------------- TAB 2: MATCH TRAJECTORY & TIMELINE ----------------
    with tab2:
        default_idx = 0
        wc_matches = df_games[
            (df_games["tournament_name"] == "FIFA_World_Cup_2022")
            & (df_games["home_name"] == "Argentina")
            & (df_games["away_name"] == "France")
        ]
        if len(wc_matches) > 0:
            default_idx = int(wc_matches.index[0])

        sel_match_label = st.selectbox("Select Match to Inspect", df_games["match_label"].tolist(), index=default_idx)
        sel_game = df_games[df_games["match_label"] == sel_match_label].iloc[0]
        gid = int(sel_game["game_id"])
        h_id, a_id = sel_game["home_team_id"], sel_game["away_team_id"]
        h_name, a_name = sel_game["home_name"], sel_game["away_name"]

        m_df = load_match_actions(gid)
        m_reg = m_df[m_df["period_id"] <= 4].copy()

        c1, c2 = st.columns([1.1, 1.1])
        with c1:
            val_metric = st.radio(
                "Select Action Valuation Metric for Timeline & Pitch:",
                ["Standard VAEP (XGBoost)", "Continuous xT (Gaussian Smoothed)"],
                horizontal=True,
            )
            v_col = "vaep_value_xgb" if "VAEP" in val_metric else "xt_continuous"

            fig_tl, ax_tl = plt.subplots(figsize=(7.5, 4.8), dpi=180)
            for tid, tname, col in [(h_id, h_name, "#2563EB"), (a_id, a_name, "#DC2626")]:
                t_sub = m_reg[m_reg["team_id"] == tid]
                cum_v = t_sub[v_col].cumsum()
                ax_tl.step(t_sub["match_minute"], cum_v, where="post", label=tname, color=col, linewidth=2.2)

            ax_tl.set_title(f"Cumulative {val_metric} Timeline", fontsize=11, weight="bold")
            ax_tl.set_xlabel("Match Minute", fontsize=9.5, weight="bold")
            ax_tl.set_ylabel(f"Cumulative {v_col}", fontsize=9.5, weight="bold")
            ax_tl.grid(True, linestyle="--", alpha=0.5)
            ax_tl.legend()
            st.pyplot(fig_tl)

        with c2:
            top_n_arrows = st.slider("Number of Top Valued Actions to Render on Pitch", 5, 40, 15)
            top_acts = m_reg.nlargest(top_n_arrows, v_col)

            fig_p, ax_p = plt.subplots(figsize=(7.5, 4.8), dpi=180)
            pitch = Pitch(pitch_type="custom", pitch_length=105, pitch_width=68, pitch_color="#0F172A", line_color="#CBD5E1")
            pitch.draw(ax=ax_p)

            for _, r in top_acts.iterrows():
                col = "#38BDF8" if r["team_id"] == h_id else "#F87171"
                pitch.arrows(
                    r["start_x"], r["start_y"], r["end_x"], r["end_y"],
                    width=2.0, headwidth=4.2, headlength=4.2, color=col, ax=ax_p, zorder=3
                )
                ax_p.text(
                    r["start_x"], r["start_y"] - 1.8,
                    f"{r['player_name'].split()[-1]} (+{r[v_col]:.2f})",
                    color="white", fontsize=7, weight="bold", ha="center"
                )
            ax_p.set_title(f"Pitch Trajectories: Top {top_n_arrows} Actions by {val_metric} (Attacking L -> R)", fontsize=10.5, weight="bold")
            st.pyplot(fig_p)

        st.markdown("#### Top High-Value Actions in Selected Match")
        st.dataframe(
            top_acts[["match_minute", "team_name", "player_name", "type_name", "result_name", "xt_continuous", "offensive_value_xgb", "defensive_value_xgb", "vaep_value_xgb"]].reset_index(drop=True),
            width="stretch",
        )

    # ---------------- TAB 3: PASS-THREAT CALCULATOR ----------------
    with tab3:
        st.subheader("Real-Time Spatial Pass-Threat (ΔxT) Calculator: Discrete Grid vs. Continuous Gaussian Surface")
        c_left, c_right = st.columns([1, 1.8])
        with c_left:
            sx = st.slider("Start X (0 = Own Goal Line, 105 = Opponent Goal Line)", 0.0, 105.0, 62.0, 0.5)
            sy = st.slider("Start Y (0 = Right Touchline, 68 = Left Touchline)", 0.0, 68.0, 18.0, 0.5)
            ex = st.slider("End X (Destination X)", 0.0, 105.0, 92.0, 0.5)
            ey = st.slider("End Y (Destination Y)", 0.0, 68.0, 34.0, 0.5)

            s_col = min(int(sx / FIELD_LENGTH * 16), 15)
            s_row = min(int(sy / FIELD_WIDTH * 12), 11)
            e_col = min(int(ex / FIELD_LENGTH * 16), 15)
            e_row = min(int(ey / FIELD_WIDTH * 12), 11)

            disc_start = float(xt_disc[s_row, s_col])
            disc_end = float(xt_disc[e_row, e_col])
            disc_delta = disc_end - disc_start

            cont_start = float(cont_interp([[sy, sx]])[0])
            cont_end = float(cont_interp([[ey, ex]])[0])
            cont_delta = cont_end - cont_start

            st.metric("Continuous Smoothed ΔxT (Ours)", f"{cont_delta:+.5f}", f"Start: {cont_start:.4f} -> End: {cont_end:.4f}")
            st.metric("Standard 16x12 Discrete ΔxT", f"{disc_delta:+.5f}", f"Start Cell ({s_col},{s_row}): {disc_start:.4f} -> End Cell ({e_col},{e_row}): {disc_end:.4f}")

        with c_right:
            fig_calc, ax_calc = plt.subplots(figsize=(8.5, 5.2), dpi=180)
            pitch = Pitch(pitch_type="custom", pitch_length=105, pitch_width=68, line_zorder=2, line_color="#1E293B")
            pitch.draw(ax=ax_calc)

            fine_y = np.linspace(0, FIELD_WIDTH, 136)
            fine_x = np.linspace(0, FIELD_LENGTH, 210)
            gy, gx = np.meshgrid(fine_y, fine_x, indexing="ij")
            fine_surf = cont_interp(np.column_stack([gy.ravel(), gx.ravel()])).reshape(136, 210)

            im = ax_calc.imshow(fine_surf, extent=[0, 105, 0, 68], origin="lower", cmap="inferno", alpha=0.88, aspect="auto")
            pitch.arrows(sx, sy, ex, ey, width=2.8, headwidth=5.5, headlength=5.5, color="#38BDF8", ax=ax_calc, zorder=4)
            ax_calc.scatter([sx, ex], [sy, ey], c=["#FFFFFF", "#38BDF8"], edgecolors="black", s=90, zorder=5)
            ax_calc.set_title(f"Pass Trajectory Vector: ({sx:.1f}, {sy:.1f}) -> ({ex:.1f}, {ey:.1f}) | Continuous ΔxT = {cont_delta:+.4f}", fontsize=11, weight="bold")
            plt.colorbar(im, ax=ax_calc, fraction=0.035, pad=0.03, label="Continuous xT Value")
            st.pyplot(fig_calc)


if __name__ == "__main__":
    main()