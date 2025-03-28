from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pandas as pd


def preprocess(game_data):
    """
    """
    # label encode precip and sky
    for col in ["precip", "sky"]:
        le = LabelEncoder()
        game_data[col] = le.fit_transform(game_data[col])

    # use one-hot encoding for team, opponent, and stadium
    game_data = pd.get_dummies(game_data, columns=["team", "opponent", "stadium"])

    # drop unnecessary columns
    game_data.drop(columns=["day_of_week_name", "date"], inplace=True)

    # scale numerical features
    scaler = StandardScaler()
    num_cols = ["division_rank", "games_behind", "cLI", "streak", "runs_scored_pg", "runs_allowed_pg",
                "runs_scored_last_10", "runs_allowed_last_10", "last_10_win_pct", "temp", "windspeed",
                "capacity", "win_pct"]
    game_data[num_cols] = scaler.fit_transform(game_data[num_cols])

    return game_data

    
