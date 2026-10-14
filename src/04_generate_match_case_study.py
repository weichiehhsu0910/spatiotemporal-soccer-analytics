import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Pitch

warnings.filterwarnings("ignore")

DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def generate_case_study_figure():
    print("[INFO] Loading match, team, player, and valued action tables...")
    df_games = pd.read_parquet(os.path.join(DATA_DIR, "games.parquet"))
    df_teams = pd.read_parquet(os.path.join(DATA_DIR, "teams.parquet"))
    df_players = pd.read_parquet(os.path.join(DATA_DIR, "players.parquet"))
    df_spadl = pd.read_parquet(os.path.join(DATA_DIR, "spadl_actions_valued.parquet"))

    team_map = dict(zip(df_teams["team_id"], df_teams["team_name"]))
    player_map = df_players.groupby("player_id")["player_name"].first().to_dict()

    # Select the 2022 FIFA World Cup Final: Argentina vs. France
    df_games["home_name"] = df_games["home_team_id"].map(team_map)
    df_games["away_name"] = df_games["away_team_id"].map(team_map)

    wc_final = df_games[
        (df_games["tournament_name"] == "FIFA_World_Cup_2022")
        & (df_games["home_name"].isin(["Argentina", "France"]))
        & (df_games["away_name"].isin(["Argentina", "France"]))
    ]
    target_game = wc_final.iloc[0] if len(wc_final) > 0 else df_games.iloc[0]
    gid = target_game["game_id"]
    h_id, a_id = target_game["home_team_id"], target_game["away_team_id"]
    h_name, a_name = team_map.get(h_id, "Home"), team_map.get(a_id, "Away")

    print(f"[INFO] Generating Case Study for Match: {h_name} vs. {a_name} (game_id={gid})...")
    m_actions = df_spadl[df_spadl["game_id"] == gid].copy().reset_index(drop=True)
    m_actions["player_name"] = m_actions["player_id"].map(player_map).fillna("Unknown")
    m_actions["team_name"] = m_actions["team_id"].map(team_map)

    # Compute continuous match minute across regular and extra time periods
    period_offsets = {1: 0.0, 2: 45.0, 3: 90.0, 4: 105.0, 5: 120.0}
    m_actions["match_minute"] = (
        m_actions["period_id"].map(period_offsets).fillna(0.0) + m_actions["time_seconds"] / 60.0
    )
    # Exclude penalty shootouts (period_id == 5) to visualize 120-minute open/set-piece dynamics
    m_reg = m_actions[m_actions["period_id"] <= 4].copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300)

    # Left panel: Cumulative VAEP momentum timeline for both teams
    ax0 = axes[0]
    for tid, tname, col in [(h_id, h_name, "#2563EB"), (a_id, a_name, "#DC2626")]:
        t_df = m_reg[m_reg["team_id"] == tid].sort_values("action_id")
        cum_vaep = t_df["vaep_value_xgb"].cumsum()
        ax0.step(t_df["match_minute"], cum_vaep, where="post", label=f"{tname} (Cum. VAEP)", color=col, linewidth=2.2)

        # Annotate goal-scoring events
        goals = t_df[(t_df["type_name"].isin(["shot", "shot_penalty", "shot_freekick"])) & (t_df["result_name"] == "goal")]
        for _, g_row in goals.iterrows():
            g_min = g_row["match_minute"]
            g_val = cum_vaep.loc[g_row.name]
            p_short = g_row["player_name"].split()[-1]
            ax0.scatter(g_min, g_val, color=col, edgecolor="black", s=70, zorder=5)
            ax0.annotate(f"Goal: {p_short}\n({int(g_min)}')", (g_min, g_val), fontsize=8, weight="bold", xytext=(0, 10), textcoords="offset points", ha="center")

    ax0.set_title(f"A. Cumulative VAEP Momentum Timeline: {h_name} vs. {a_name}", fontsize=12, weight="bold")
    ax0.set_xlabel("Match Minute (Regular + Extra Time)", fontsize=10, weight="bold")
    ax0.set_ylabel("Cumulative VAEP Value (XGBoost)", fontsize=10, weight="bold")
    ax0.grid(True, linestyle="--", alpha=0.5)
    ax0.legend(loc="upper left", fontsize=9.5)

    # Right panel: Pitch trajectory renderer for top 16 progressive actions by Continuous xT
    ax1 = axes[1]
    pitch = Pitch(pitch_type="custom", pitch_length=105, pitch_width=68, pitch_color="#0F172A", line_color="#94A3B8", line_zorder=2)
    pitch.draw(ax=ax1)

    move_types = ["pass", "dribble", "cross", "through_ball"]
    top_moves = (
        m_reg[m_reg["type_name"].isin(move_types) & (m_reg["result_name"] == "success")]
        .nlargest(16, "xt_continuous")
        .copy()
    )

    for _, row in top_moves.iterrows():
        col = "#38BDF8" if row["team_id"] == h_id else "#F87171"
        pitch.arrows(
            row["start_x"], row["start_y"], row["end_x"], row["end_y"],
            width=2.2, headwidth=4.5, headlength=4.5, color=col, ax=ax1, zorder=3, alpha=0.9
        )
        p_short = row["player_name"].split()[-1]
        ax1.text(
            row["start_x"], row["start_y"] - 1.8,
            f"{p_short} (+{row['xt_continuous']:.2f} xT)",
            fontsize=7, color="white", weight="bold", ha="center", zorder=4
        )

    ax1.set_title(f"B. Pitch Trajectory Renderer: Top 16 High-Threat (ΔxT) Actions (L -> R)", fontsize=12, weight="bold")

    plt.suptitle(f"Match Case Study: 2022 FIFA World Cup Final ({h_name} vs. {a_name}) — Action Valuation & Spatial Trajectories", fontsize=13.5, weight="bold", y=0.99)
    plt.tight_layout()
    fig3_path = os.path.join(MODEL_DIR, "fig3_world_cup_final_case_study.png")
    plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved Case Study Figure to: {fig3_path}")


if __name__ == "__main__":
    generate_case_study_figure()