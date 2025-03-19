import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import random


def get_teams():
    """ Gets all team names from the stadium_capacity.csv file (all team codes from 2000-present). """
    teams = pd.read_csv('data/stadium_capacity.csv')
    return teams.Team.unique()


def get_game_info(teams):
    """ 
        Takes an array of team names (3 letter strings) and scrapes baseball reference for game data 
        for all teams from 2000-2024 excluding 2020 and 2021.
    """
    all_columns = ["date", "boxscore", "team", "@", "opponent", "w_or_l",
               "runs_scored", "runs_allowed", "innings", "record",
               "division_rank", "games_behind", "winning_pitcher",
               "losing_pitcher", "save", "time", "day_or_night",
               "attendance", "cLI", "streak", "orig_scheduled"]
    df_all_games = pd.DataFrame(columns=all_columns)

    # all years from 2000 to 2024 (excluding 2020 and 2021 due to COVID attendance restrictions)
    years = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2022, 2023, 2024]

    for team in teams:
        for year in years:
            print(f"Scraping {year} {team}...")

            try:
                URL = f"https://www.baseball-reference.com/teams/{team}/{year}-schedule-scores.shtml"
                res = requests.get(URL)
                res.encoding = 'utf-8'
                soup = BeautifulSoup(res.text, 'html.parser')

                columns = ["date", "boxscore", "team", "@", "opponent", "w_or_l",
                    "runs_scored", "runs_allowed", "innings", "record",
                    "division_rank", "games_behind", "winning_pitcher",
                    "losing_pitcher", "save", "time", "day_or_night",
                    "attendance", "cLI", "streak", "orig_scheduled"]
                df_team = pd.DataFrame(columns=columns)

                # create rows in df_team for each game from rows in the stats table
                game_table = soup.find("table", {"class": "stats_table"})
                games = game_table.find_all('tr')
                for game in games:
                    data = []
                    cols = game.find_all('td')
                    for col in cols:
                        data.append(col.text)
                    
                    if len(data) == len(columns):
                        df_team.loc[len(df_team)] = data 

                # add binary dummy variable for double headers (1 = part of dh, 0 = not part of dh)
                df_team['dh'] = [1 if "(" in date else 0 for date in df_team["date"]]
                
                # convert date column to datetime
                df_team["date"] = df_team["date"].str.replace(r"\(.*\)", "", regex=True).str.strip()
                df_team["date"] = df_team["date"].apply(lambda x: f"{x.split(', ')[1]}, {year}" if ', ' in x else x)
                df_team["date"] = pd.to_datetime(df_team["date"], errors='coerce')
                
                # shift division_rank, streak, games_behind, and record columns since we
                # are interested in what these values are going into each game, not after
                df_team["division_rank"] = df_team["division_rank"].shift().fillna("0")
                df_team["streak"] = df_team["streak"].shift().fillna("")
                df_team["games_behind"] = df_team["games_behind"].shift().fillna("0")
                df_team["record"] = df_team["record"].shift().fillna("0-0")

                # convert runs_scored and runs_allowed to numeric values
                df_team["runs_scored"] = pd.to_numeric(df_team["runs_scored"], errors='coerce')
                df_team["runs_allowed"] = pd.to_numeric(df_team["runs_allowed"], errors='coerce')  

                # add columns with rolling means for runs scored and runs allowed
                df_team["runs_scored_pg"] = df_team["runs_scored"].expanding().mean().shift().fillna(0)
                df_team["runs_allowed_pg"] = df_team["runs_allowed"].expanding().mean().shift().fillna(0)

                # add columns with rolling means for runs scored and runs allowed over the last 10 games
                df_team["runs_scored_last_10"] = df_team["runs_scored"].rolling(window=10, min_periods=1).mean().shift().fillna(0)
                df_team["runs_allowed_last_10"] = df_team["runs_allowed"].rolling(window=10, min_periods=1).mean().shift().fillna(0)
                
                # add column with winning percentage over the last 10 games
                df_team['win'] = [1 if "W" in res else 0 for res in df_team["w_or_l"]]
                df_team["last_10_win_pct"] = df_team["win"].rolling(window=10, min_periods=1).mean().shift().fillna(0)
            
                # keep only home games
                df_team = df_team[df_team['@'].str.contains('@') == False].reset_index(drop=True)

                # add binary dummy variable for whether the game was on opening day (1 = yes, 0 = no)
                df_team['opening_day'] = [1 if date == min(df_team["date"]) else 0 for date in df_team["date"]]

                # combine df_team and df_all_games
                df_all_games = pd.concat([df_all_games, df_team], ignore_index=True)

                time.sleep(random.uniform(1.5, 2.5))
            except Exception as e:
                print(f"Error scraping {year} {team}:", e)
            
    df_all_games.to_csv('data/game_data.csv', index=False, encoding='utf-8')
    return df_all_games


def main():
    """ Call get_game_info() to scrape baseball reference for game data for all teams (from get_teams()) from 2000-2024. """
    teams = get_teams()
    start_time = time.time()
    get_game_info(teams)
    end_time = time.time()
    print("Execution time:", end_time - start_time)


if __name__ == "__main__":
    main()
