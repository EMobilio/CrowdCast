import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import imageio
import numpy as np
import calendar
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.preprocess import preprocess


def get_summary(game_data):
    """
        Takes a DataFrame with game data and a boolean and prints some summary statistics on the data.
    """
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", "{:.2f}".format) 
    print("Summary Statistics:\n", game_data.describe(include="all"))
    print("\nMissing Values:\n", game_data.isnull().sum())
    print("\nData Types:\n", game_data.dtypes)


def generate_corr_matrix(game_data, save):
    """
        Takes a DataFrame with game data and a boolean and generates a correlation matrix of the features.
    """
    corr_matrix = game_data.corr(numeric_only=True)
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix of Numerical Features")
    plt.tight_layout()
    
    if save:
        plt.savefig("plots/corr_matrix.png", dpi=300, bbox_inches="tight")


def generate_reg_plots(game_data, save):
    """
        Takes a DataFrame with game data and generates some regression plots of attendance vs. some quantitative variables
    """
    # Attendance vs. Win Percentage 
    plt.figure(figsize=(10, 5))
    sns.regplot(x="win_pct", y="attendance", data=game_data.sample(frac=0.1))
    plt.xlabel("Win Percentage")
    plt.ylabel("Attendance")
    plt.title("Win Percentage vs. Attendance")

    if save:
        plt.savefig("plots/regplots/att_vs_win_pct.png", dpi=300, bbox_inches="tight")

    # Attendance vs. Runs Scored/Game
    plt.figure(figsize=(10, 5))
    sns.regplot(x="runs_scored_pg", y="attendance", data=game_data.sample(frac=0.1))
    plt.xlabel("Runs Scored Per Game")
    plt.ylabel("Attendance")
    plt.title("Runs Scored Per Game vs. Attendance")

    if save:
        plt.savefig("plots/regplots/att_vs_runs_pg.png", dpi=300, bbox_inches="tight")

    # Attendance vs.temperature
    plt.figure(figsize=(10, 5))
    sns.regplot(x="temp", y="attendance", data=game_data.sample(frac=0.1))
    plt.xlabel("Temperature (°F)")
    plt.ylabel("Attendance")
    plt.title("Temperature vs. Attendance")

    if save:
        plt.savefig("plots/regplots/att_vs_temp.png", dpi=300, bbox_inches="tight")


def generate_boxplots(game_data, save):
    """
        Takes a DataFrame with game data and a boolean and generates boxplots showing attendance by some categorical variables.
    """
    # Attendance by precipitation
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=game_data['precip'], y=game_data['attendance'], palette=None, hue=game_data['precip'])
    plt.xlabel("Precipitation")
    plt.ylabel("Attendance")
    plt.title("Attendance by Precipitation")
    plt.xticks(rotation=45)

    if save:
        plt.savefig("plots/boxplots/att_by_precip.png", dpi=300, bbox_inches="tight")

    # Attendance by sky description 
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=game_data['sky'], y=game_data['attendance'], palette=None, hue=game_data['sky'])
    plt.xlabel("Sky Condition")
    plt.ylabel("Attendance")
    plt.title("Attendance by Sky Condition")
    plt.xticks(rotation=45)

    if save:
        plt.savefig("plots/boxplots/att_by_sky.png", dpi=300, bbox_inches="tight")

    # Attendance by day of the week
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=game_data['day_of_week_name'], y=game_data['attendance'], 
                order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], 
                palette=None, hue=game_data['day_of_week_name'])
    plt.xlabel("Day of the Week")
    plt.ylabel("Attendance")
    plt.title("Attendance by Day of the Week")
    plt.xticks(rotation=45)

    if save:
        plt.savefig("plots/boxplots/att_by_weekday.png", dpi=300, bbox_inches="tight")

    # Attendance by Year
    plt.figure(figsize=(12, 5))
    sns.boxplot(x=game_data['year'], y=game_data['attendance'], palette=None, hue=game_data['year'].astype(str))
    plt.xlabel("Year")
    plt.ylabel("Attendance")
    plt.title("Attendance by Year")
    plt.xticks(rotation=45)

    if save:
        plt.savefig("plots/boxplots/att_by_year.png", dpi=300, bbox_inches="tight")

    # Attendance by Month
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=game_data['month'], y=game_data['attendance'], palette=None, hue=game_data['month'].astype(str))
    plt.xlabel("Month")
    plt.ylabel("Attendance")
    plt.title("Attendance by Month")
    plt.xticks(ticks=range(8), labels=[calendar.month_name[i] for i in range(3, 11)], rotation=45)
    
    if save:
        plt.savefig("plots/boxplots/att_by_month.png", dpi=300, bbox_inches="tight")

    # Attendance by Team
    plt.figure(figsize=(12, 5))
    sns.boxplot(x=game_data['team'], y=game_data['attendance'], palette=None, hue=game_data['team'])
    plt.xlabel("Team")
    plt.ylabel("Attendance")
    plt.title("Attendance by Team")
    plt.xticks(rotation=45)

    if save:
        plt.savefig("plots/boxplots/att_by_team.png", dpi=300, bbox_inches="tight")


def generate_tsne_3d_gif(X, y, perplexity=30, filename="tsne_3d.gif"):
    """
        Generates a rotating 3D t-SNE scatter plot of the dataset and saves it as a GIF.
    """
    tsne = TSNE(n_components=3, perplexity=perplexity, random_state=42)
    X_embedded = tsne.fit_transform(X)

    images = []
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    norm = plt.Normalize(vmin=np.min(y), vmax=np.max(y))
    cmap = plt.get_cmap('viridis')

    for angle in range(0, 360, 5):
        ax.clear()
        scatter = ax.scatter(
            X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2],
            c=y, cmap=cmap, norm=norm, s=10
        )
        ax.set_title("3D t-SNE: Attendance Visualization")
        ax.view_init(30, angle)

        # Add colorbar
        if angle == 0:
            mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            mappable.set_array([])
            cbar = plt.colorbar(mappable, ax=ax, shrink=0.6)
            cbar.set_label("Attendance")

        plt.tight_layout()

        fname = f"frame_{angle}.png"
        plt.savefig(fname, dpi=100)
        images.append(imageio.imread(fname))

    os.makedirs("plots", exist_ok=True)
    imageio.mimsave(f"plots/{filename}", images, fps=10)

    for fname in [f"frame_{a}.png" for a in range(0, 360, 5)]:
        os.remove(fname)


def main():
    """
        Generates summary stats and some plots for EDA of the data.
    """
    game_data = pd.read_csv("data/MLB_games_2000-2024.csv")

    # get summary and generate plots
    get_summary(game_data)
    generate_corr_matrix(game_data, save=True)
    generate_reg_plots(game_data, save=True)
    generate_boxplots(game_data, save=True)
    X = preprocess(game_data.drop(columns=["attendance"]), model="XGBoost", should_scale=False)
    generate_tsne_3d_gif(X, game_data["attendance"], filename="attendance_tsne.gif")


if __name__ == "__main__":
    main()