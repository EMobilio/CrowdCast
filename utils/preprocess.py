import pandas as pd
import numpy as np


def preprocess(game_data, model):
    """
        Takes a DataFrame containing game data and a string and performs encoding and scaling of the data,
        returning the processed DataFrame.
    """
    # use one-hot encoding for team, opponent, and stadium
    game_data = pd.get_dummies(game_data, columns=["team", "opponent", "stadium", "precip", "sky"])

    # drop unnecessary columns
    game_data.drop(columns=["day_of_week_name", "date"], inplace=True)

    # cyclic encoding for 'day_of_week' (if using linear model)
    if model == "linear":
        game_data["day_of_week_sin"] = np.sin(2 * np.pi * game_data["day_of_week"] / 7)
        game_data["day_of_week_cos"] = np.cos(2 * np.pi * game_data["day_of_week"] / 7)

    return game_data