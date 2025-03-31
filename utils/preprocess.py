import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def preprocess(game_data, model):
    """
        Takes a DataFrame containing game data and a string and performs encoding and scaling of the data,
        returning the processed DataFrame.
    """
    # drop unnecessary columns
    game_data.drop(columns=["day_of_week_name", "date"], inplace=True)

    # use one-hot encoding for precip, sky, team
    game_data = pd.get_dummies(game_data, columns=["precip", "sky", "team", "opponent"])

    # cyclic encoding for 'day_of_week' (if using linear model)
    if model == "linear":
        game_data["day_of_week_sin"] = np.sin(2 * np.pi * game_data["day_of_week"] / 7)
        game_data["day_of_week_cos"] = np.cos(2 * np.pi * game_data["day_of_week"] / 7)

        game_data["month_sin"] = np.sin(2 * np.pi * game_data["month"] / 12)
        game_data["month_cos"] = np.cos(2 * np.pi * game_data["month"] / 12)

    scaler = StandardScaler()
    game_data = scaler.fit_transform(game_data)

    return game_data