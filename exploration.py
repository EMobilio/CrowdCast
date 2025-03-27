import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def get_summary(game_data):
    """
    """
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", "{:.2f}".format) 
    print("Summary Statistics:\n", game_data.describe(include="all"))
    print("\nMissing Values:\n", game_data.isnull().sum())
    print("\nData Types:\n", game_data.dtypes)


def generate_corr_matrix(game_data):
    """
    """
    corr_matrix = game_data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix of Numerical Features")
    plt.tight_layout()
    plt.show()


def main():
    """
    """
    game_data = pd.read_csv("data/MLB_games_2000-2024.csv")

    get_summary(game_data)
    generate_corr_matrix(game_data)


if __name__ == "__main__":
    main()