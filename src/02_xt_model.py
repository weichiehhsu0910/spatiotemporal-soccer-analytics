import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
import socceraction.spadl as spadl
import socceraction.xthreat as xthreat

warnings.filterwarnings("ignore")

DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

FIELD_LENGTH = 105.0
FIELD_WIDTH = 68.0

def build_continuous_interpolator(xt_surface: np.ndarray, sigma: float = 1.0):
    """
    Apply 2D Gaussian kernel smoothing to the discrete xT matrix and construct
    a continuous 2D bilinear interpolator over the pitch coordinates.
    xt_surface shape: (w, l) = (12, 16) -> (y-axis width, x-axis length)
    """
    # 1. 2D Gaussian kernel smoothing
    xt_smoothed = gaussian_filter(xt_surface, sigma=sigma, mode="nearest")

    n_y, n_x = xt_smoothed.shape
    # Compute spatial cell centroids across pitch coordinates (y: 0~68m, x: 0~105m)
    y_centers = np.linspace(FIELD_WIDTH / (2 * n_y), FIELD_WIDTH - FIELD_WIDTH / (2 * n_y), n_y)
    x_centers = np.linspace(FIELD_LENGTH / (2 * n_x), FIELD_LENGTH - FIELD_LENGTH / (2 * n_x), n_x)

    # # 2. Construct 2D continuous bilinear interpolator with boundary extrapolation
    interpolator = RegularGridInterpolator(
        (y_centers, x_centers),
        xt_smoothed,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    return xt_smoothed, interpolator

def run_xt_pipeline():
    print("[INFO] Loading SPADL actions, players, and teams...")
    df_spadl = pd.read_parquet(os.path.join(DATA_DIR, "spadl_actions.parquet"))
    df_players = pd.read_parquet(os.path.join(DATA_DIR, "players.parquet"))
    df_teams = pd.read_parquet(os.path.join(DATA_DIR, "teams.parquet"))
    df_games = pd.read_parquet(os.path.join(DATA_DIR, "games.parquet"))

    # 1. Standardize play direction for all matches from left to right (x: 0 -> 105)
    print("[INFO] Standardizing play direction (Left-to-Right) across 166 matches...")
    ltr_list = []
    for game_id, group in df_spadl.groupby("game_id", sort=False):
        home_team_id = group["home_team_id"].iloc[0]
        group_ltr = spadl.play_left_to_right(group, home_team_id=home_team_id)
        ltr_list.append(group_ltr)
    df_actions_ltr = pd.concat(ltr_list, ignore_index=True)

    # 2. Fit standard 16x12 discrete Markov Decision Process (MDP) xT model via value iteration
    print("[INFO] Fitting Markov Decision Process (MDP) Expected Threat (xT) model...")
    xt_model = xthreat.ExpectedThreat(l=16, w=12, eps=1e-5)
    xt_model.fit(df_actions_ltr)

    xt_discrete_matrix = xt_model.xT
    num_iterations = len(xt_model.heatmaps)

    # 3. Apply 2D Gaussian smoothing and construct continuous spatial interpolator
    print("[INFO] Applying 2D Gaussian Kernel Smoothing (sigma=0.85) & Bilinear Interpolation...")
    xt_smoothed_matrix, cont_interp = build_continuous_interpolator(xt_discrete_matrix, sigma=0.85)

    # Save discrete and smoothed xT surface matrices
    np.save(os.path.join(MODEL_DIR, "xt_discrete_matrix.npy"), xt_discrete_matrix)
    np.save(os.path.join(MODEL_DIR, "xt_smoothed_matrix.npy"), xt_smoothed_matrix)

    # 4. Evaluate discrete and continuous xT across all successful ball-progressing actions
    print("[INFO] Rating ball-progressing actions (Discrete vs. Continuous xT)...")
    df_mov = xthreat.get_successful_move_actions(df_actions_ltr).copy()

    # 4a. Discrete grid xT valuation
    df_mov["xt_discrete"] = xt_model.rate(df_mov, use_interpolation=False)

    # 4b. Continuous smoothed xT valuation
    start_pts = np.column_stack([
        np.clip(df_mov["start_y"].values, 0.0, FIELD_WIDTH),
        np.clip(df_mov["start_x"].values, 0.0, FIELD_LENGTH),
    ])
    end_pts = np.column_stack([
        np.clip(df_mov["end_y"].values, 0.0, FIELD_WIDTH),
        np.clip(df_mov["end_x"].values, 0.0, FIELD_LENGTH),
    ])
    df_mov["xt_start_cont"] = cont_interp(start_pts)
    df_mov["xt_end_cont"] = cont_interp(end_pts)
    df_mov["xt_continuous"] = df_mov["xt_end_cont"] - df_mov["xt_start_cont"]

    # Merge ratings back into the main SPADL DataFrame (non-move actions receive 0.0 xT)
    df_actions_ltr["xt_discrete"] = 0.0
    df_actions_ltr["xt_continuous"] = 0.0
    df_actions_ltr.loc[df_mov.index, "xt_discrete"] = df_mov["xt_discrete"]
    df_actions_ltr.loc[df_mov.index, "xt_continuous"] = df_mov["xt_continuous"]

    # Save SPADL actions enriched with xT values
    df_actions_ltr.to_parquet(os.path.join(DATA_DIR, "spadl_actions_with_xt.parquet"), index=False)

    # 5. Generate publication figure comparing discrete vs. continuous Gaussian-smoothed xT surfaces
    print("[INFO] Generating publication figure: Discrete vs. Continuous xT Surface...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    pitch = Pitch(pitch_type="custom", pitch_length=105, pitch_width=68, line_zorder=2, line_color="#1E293B")

    # Left panel: 16x12 discrete grid
    pitch.draw(ax=axes[0])
    im0 = axes[0].imshow(
        xt_discrete_matrix,
        extent=[0, 105, 0, 68],
        origin="lower",
        cmap="inferno",
        alpha=0.88,
        aspect="auto",
    )
    axes[0].set_title("A. Standard Discrete xT Surface (16x12 Grid, Value Iteration)", fontsize=12, weight="bold")
    plt.colorbar(im0, ax=axes[0], fraction=0.035, pad=0.03, label="Expected Threat (xT)")

    # Right panel: Continuous Gaussian-smoothed surface rendered over a 210x136 fine mesh
    pitch.draw(ax=axes[1])
    fine_y = np.linspace(0, FIELD_WIDTH, 136)
    fine_x = np.linspace(0, FIELD_LENGTH, 210)
    grid_y, grid_x = np.meshgrid(fine_y, fine_x, indexing="ij")
    fine_pts = np.column_stack([grid_y.ravel(), grid_x.ravel()])
    fine_surface = cont_interp(fine_pts).reshape(136, 210)

    im1 = axes[1].imshow(
        fine_surface,
        extent=[0, 105, 0, 68],
        origin="lower",
        cmap="inferno",
        alpha=0.88,
        aspect="auto",
    )
    axes[1].set_title("B. Continuous Gaussian-Smoothed xT Surface (2D Bilinear Interpolated)", fontsize=12, weight="bold")
    plt.colorbar(im1, ax=axes[1], fraction=0.035, pad=0.03, label="Continuous Expected Threat (xT)")

    plt.suptitle("Expected Threat (xT) Valuation Surface: Discrete Grid vs. Continuous Gaussian Smoothing (Attacking Left -> Right)", fontsize=14, weight="bold", y=0.98)
    plt.tight_layout()
    fig_path = os.path.join(MODEL_DIR, "fig1_xt_surface_comparison.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 6. Aggregate cumulative and per-90 player xT leaderboards
    player_minutes = df_players.groupby(["player_id", "player_name"])["minutes_played"].sum().reset_index()
    
    player_minutes = (
        df_players.groupby("player_id")
        .agg(player_name=("player_name", "first"), total_minutes=("minutes_played", "sum"))
        .reset_index()
    )

    player_xt = (
        df_mov.groupby("player_id")
        .agg(
            move_actions_count=("action_id", "count"),
            total_xt_discrete=("xt_discrete", "sum"),
            total_xt_continuous=("xt_continuous", "sum"),
            mean_xt_continuous=("xt_continuous", "mean"),
        )
        .reset_index()
    )
    player_xt = player_xt.merge(player_minutes, on="player_id", how="left")
    player_xt["xt_cont_per90"] = (player_xt["total_xt_continuous"] / player_xt["total_minutes"].replace(0, np.nan)) * 90.0

    top_total = player_xt.sort_values(by="total_xt_continuous", ascending=False).head(10)
    top_per90 = player_xt[player_xt["total_minutes"] >= 450].sort_values(by="xt_cont_per90", ascending=False).head(10)

    corr_val = np.corrcoef(df_mov["xt_discrete"], df_mov["xt_continuous"])[0, 1]

    print("\n" + "=" * 70)
    print("STAGE 2: CONTINUOUS xT MODEL COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"MDP Value Iteration Convergence Steps : {num_iterations}")
    print(f"Discrete xT Surface Min / Max         : {xt_discrete_matrix.min():.5f} / {xt_discrete_matrix.max():.5f}")
    print(f"Smoothed xT Surface Min / Max         : {xt_smoothed_matrix.min():.5f} / {xt_smoothed_matrix.max():.5f}")
    print(f"Total Successful Move Actions Rated   : {len(df_mov):,}")
    print(f"Pearson r (Discrete vs Continuous xT) : {corr_val:.5f}")
    print(f"Saved Surface Figure                  : {fig_path}")
    print("-" * 70)
    print("TOP 10 PLAYERS BY CUMULATIVE CONTINUOUS xT:")
    print(top_total[["player_name", "total_minutes", "move_actions_count", "total_xt_discrete", "total_xt_continuous", "xt_cont_per90"]].to_string(index=False))
    print("-" * 70)
    print("TOP 10 PLAYERS BY CONTINUOUS xT PER 90 MINS (Min. 450 mins played):")
    print(top_per90[["player_name", "total_minutes", "move_actions_count", "total_xt_continuous", "xt_cont_per90"]].to_string(index=False))
    print("=" * 70)

if __name__ == "__main__":
    run_xt_pipeline()