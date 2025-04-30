import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


def preprocess(game_data, model, should_scale=True):
    """
        Takes a DataFrame containing game data and a string and performs encoding and scaling of the data,
        returning the processed DataFrame.
    """
    # drop unnecessary columns
    game_data.drop(columns=["day_of_week_name", "date"], inplace=True)

    # use one-hot encoding for precip, sky
    game_data = pd.get_dummies(game_data, columns=["precip", "sky"])

    # use label encoding for team, opponent, stadium
    le = LabelEncoder()
    game_data["team_label"] = le.fit_transform(game_data["team"])
    game_data["opponent_label"] = le.fit_transform(game_data["opponent"])
    game_data["stadium_label"] = le.fit_transform(game_data["stadium"])
    game_data.drop(columns=["team", "opponent", "stadium"], inplace=True)


    # cyclic encoding for 'day_of_week' (if using linear model)
    if model == "linear":
        game_data["day_of_week_sin"] = np.sin(2 * np.pi * game_data["day_of_week"] / 7)
        game_data["day_of_week_cos"] = np.cos(2 * np.pi * game_data["day_of_week"] / 7)

        game_data["month_sin"] = np.sin(2 * np.pi * game_data["month"] / 12)
        game_data["month_cos"] = np.cos(2 * np.pi * game_data["month"] / 12)

    # scale if specified
    if should_scale:
        original_column_names = game_data.columns
        scaler = StandardScaler()
        scaled_array = scaler.fit_transform(game_data)
        game_data = pd.DataFrame(scaled_array, columns=original_column_names) 

    return game_data