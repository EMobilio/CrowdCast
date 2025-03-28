from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np


def preprocess(game_data, model):
    """
        Takes a DataFrame containing game data and a string and performs encoding and scaling of the data,
        returning the processed DataFrame.
    """
    # label encode precip and sky
    for col in ["precip", "sky"]:
        le = LabelEncoder()
        game_data[col] = le.fit_transform(game_data[col])

    # use one-hot encoding for team, opponent, and stadium
    game_data = pd.get_dummies(game_data, columns=["team", "opponent", "stadium"])

    # drop unnecessary columns
    game_data.drop(columns=["day_of_week_name", "date"], inplace=True)

    # cyclic encoding for 'ay_of_week (if using linear model)
    if model == "linear":
        game_data["day_of_week_sin"] = np.sin(2 * np.pi * game_data["day_of_week"] / 7)
        game_data["day_of_week_cos"] = np.cos(2 * np.pi * game_data["day_of_week"] / 7)

    # scale numerical features
    scaler = StandardScaler()
    num_cols = ["division_rank", "games_behind", "cLI", "streak", "runs_scored_pg", "runs_allowed_pg",
                "runs_scored_last_10", "runs_allowed_last_10", "last_10_win_pct", "temp", "windspeed",
                "capacity", "win_pct"]
    game_data[num_cols] = scaler.fit_transform(game_data[num_cols])

    return game_data