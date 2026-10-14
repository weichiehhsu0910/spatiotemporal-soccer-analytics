import os
import warnings
import pandas as pd
from tqdm import tqdm
from socceraction.data.statsbomb import StatsBombLoader
import socceraction.spadl as spadl
import socceraction.atomic.spadl as atomicspadl

warnings.filterwarnings("ignore")

# Target tournaments: (tournament_name, competition_id, season_id)
TOURNAMENTS = [
    ("FIFA_World_Cup_2022", 43, 106),
    ("UEFA_Euro_2024", 55, 282),
    ("UEFA_Euro_2020", 55, 43),
]

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def run_etl():
    # Initialize remote loader for Hudl StatsBomb Open Data repository
    SBL = StatsBombLoader(getter="remote", creds={"user": None, "passwd": None})

    all_games = []
    all_teams = []
    all_players = []
    spadl_actions_list = []
    atomic_actions_list = []
    total_raw_events = 0

    for tourney_name, comp_id, season_id in TOURNAMENTS:
        print(f"\n[INFO] Fetching games for {tourney_name} (comp_id={comp_id}, season_id={season_id})...")
        games = SBL.games(competition_id=comp_id, season_id=season_id)
        games["tournament_name"] = tourney_name
        all_games.append(games)

        for _, game in tqdm(games.iterrows(), total=len(games), desc=f"Processing {tourney_name}"):
            game_id = game["game_id"]
            home_team_id = game["home_team_id"]

            # 1. Fetch teams, players, and raw match event stream
            teams = SBL.teams(game_id=game_id)
            players = SBL.players(game_id=game_id)
            events = SBL.events(game_id=game_id)

            all_teams.append(teams)
            all_players.append(players)
            total_raw_events += len(events)

            # 2. Convert raw events into standard SPADL format (retaining relative home/away orientation for gamestates)
            actions = spadl.statsbomb.convert_to_actions(events, home_team_id=home_team_id)
            actions = spadl.add_names(actions)
            actions["game_id"] = game_id
            actions["home_team_id"] = home_team_id
            actions["tournament_name"] = tourney_name
            spadl_actions_list.append(actions)

            # 3. Convert standard SPADL actions into Atomic-SPADL format
            atomic_actions = atomicspadl.convert_to_atomic(actions)
            atomic_actions = atomicspadl.add_names(atomic_actions)
            atomic_actions["game_id"] = game_id
            atomic_actions["home_team_id"] = home_team_id
            atomic_actions["tournament_name"] = tourney_name
            atomic_actions_list.append(atomic_actions)

    # Concatenate all tables across tournaments
    df_games = pd.concat(all_games, ignore_index=True)
    df_teams = pd.concat(all_teams, ignore_index=True).drop_duplicates(subset=["team_id"])
    df_players = pd.concat(all_players, ignore_index=True)
    df_spadl = pd.concat(spadl_actions_list, ignore_index=True)
    df_atomic = pd.concat(atomic_actions_list, ignore_index=True)

    # Save processed tables as Parquet files
    df_games.to_parquet(os.path.join(DATA_DIR, "games.parquet"), index=False)
    df_teams.to_parquet(os.path.join(DATA_DIR, "teams.parquet"), index=False)
    df_players.to_parquet(os.path.join(DATA_DIR, "players.parquet"), index=False)
    df_spadl.to_parquet(os.path.join(DATA_DIR, "spadl_actions.parquet"), index=False)
    df_atomic.to_parquet(os.path.join(DATA_DIR, "atomic_spadl_actions.parquet"), index=False)

    # Print dataset summary statistics
    print("\n" + "="*60)
    print("ETL PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Total Matches Processed       : {len(df_games):,}")
    print(f"Unique Teams                  : {len(df_teams):,}")
    print(f"Unique Players                : {df_players['player_id'].nunique():,}")
    print(f"Total Raw StatsBomb Events    : {total_raw_events:,}")
    print(f"Total SPADL Actions           : {len(df_spadl):,}")
    print(f"Total Atomic-SPADL Actions    : {len(df_atomic):,}")
    print("="*60)

if __name__ == "__main__":
    run_etl()