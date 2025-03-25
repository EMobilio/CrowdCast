import pandas as pd


def update_hometeam_column(df):
    """
        Takes a DataFrame containing data from the retrosheet CSV and returns an updated DataFrame 
        with the hometeam column matching baseball-reference team names.
    """
    # mapping of retrosheet team names to baseball-reference team names
    team_mapping = {
        'ANA': 'LAA',
        'ARI': 'ARI',
        'ATL': 'ATL',
        'BAL': 'BAL',
        'BOS': 'BOS',
        'CHN': 'CHC',
        'CHA': 'CHW',
        'CIN': 'CIN',
        'CLE': 'CLE',
        'COL': 'COL',
        'DET': 'DET',
        'FLO': 'FLA',
        'HOU': 'HOU',
        'KCA': 'KCR',
        'LAN': 'LAD',
        'MIA': 'MIA',
        'MIL': 'MIL',
        'MIN': 'MIN',
        'MON': 'MON',
        'NYA': 'NYY',
        'NYN': 'NYM',
        'OAK': 'OAK',
        'PHI': 'PHI',
        'PIT': 'PIT',
        'SDN': 'SDP',
        'SEA': 'SEA',
        'SFN': 'SFG',
        'SLN': 'STL',
        'TBA': 'TBR',
        'TEX': 'TEX',
        'TOR': 'TOR',
        'WAS': 'WSN',
    }
    
    # apply mapping 
    def map_team(row):
        team = row['hometeam']
        season = row['season']
        
        # 2000-2004 ANA should still be ANA, 2000-2007 TBA should be TBD
        if team == 'ANA' and 2000 <= season <= 2004:
            return 'ANA'
        elif team == 'TBA' and 2000 <= season <= 2007:
            return 'TBD'
        
        return team_mapping.get(team, team)  # default mapping
    
    df['hometeam'] = df.apply(map_team, axis=1)
    
    return df


def merge_data():
    """
       Adds weather data and stadium from retrosheet and stadium capacity CSVs to game_data CSV records and returns the complete DataFrame.
    """
    game_data_df = pd.read_csv("data/game_data.csv")
    retrosheet_df = pd.read_csv("data/retrosheet_gameinfo_2000-2024.csv")
    stadium_df = pd.read_csv("data/stadium_capacity_2000-2024.csv")

    # call update_hometeam_column() to ensure retrosheet hometeam column matches baseball-reference team names
    retrosheet_df = update_hometeam_column(retrosheet_df)

    # ensure date columns are in the same format
    retrosheet_df['date'] = pd.to_datetime(retrosheet_df['date'], format='%Y%m%d').dt.date
    game_data_df['date'] = pd.to_datetime(game_data_df['date']).dt.date

    # to differentiate double header games, add a number column telling which game
    # in the double header it is (or 0 if not part of a dh)
    game_counts = game_data_df.groupby(["date", "team"]).size()
    game_data_df = game_data_df.merge(game_counts.rename("game_count"), on=["date", "team"])
    game_data_df["number"] = game_data_df.groupby(["date", "team"]).cumcount() + 1
    game_data_df.loc[game_data_df["game_count"] == 1, "number"] = 0.0
    game_data_df.drop(columns=["game_count"], inplace=True)
    game_data_df["number"] = game_data_df["number"].astype(float)

    # select date, hometeam, and number columns to merge on, and precip, sky, temp and windspeed columns to merge into game_data
    retrosheet_df = retrosheet_df[['date', 'hometeam', 'precip', 'sky', 'temp', 'windspeed', 'number']]
    retrosheet_df = retrosheet_df.rename(columns={'hometeam': 'team'}) # rename to match game_data

    # merge on date, team, and number
    merged_df = game_data_df.merge(retrosheet_df, left_on=['date', 'team', 'number'], 
                                right_on=['date', 'team', 'number'], how='left')
    
    # create a year column and extract columns from stadium_df
    merged_df["year"] = pd.to_datetime(merged_df["date"]).dt.year
    stadium_df = stadium_df[["Team", "Year", "Stadium", "Capacity"]]

    # merge in stadium data and drop the extra columns
    merged_df = merged_df.merge(stadium_df, left_on=['team', 'year'], right_on=["Team", "Year"], how='left')
    merged_df.rename(columns={"Stadium": "stadium", "Capacity": "capacity"}, inplace=True)
    merged_df.drop(columns=["Team", "Year"], inplace=True)

    return merged_df

def print_data_info(df):
    """
        Takes a DataFrame containing the merged game data and prints some summary information.
    """
    print("DF Shape:", df.shape)
    print()
    print("DF Shape after dropping duplicates:", df.drop_duplicates().shape)
    print()
    print("Number of missing attendance values:", df[df.attendance.isnull()].shape)
    print()
    print("Column types:")
    print(df.dtypes)
    print()
    print("Number of double header games with attendance data:", df[((df.attendance.notnull()) & (df.dh == 1))].shape)
    print()
    print("Number of nulls for each column:")
    print(df.isnull().sum())
    print()
    print("precip value counts:")
    print(df["precip"].value_counts(dropna=False))
    print()
    print("sky value counts:")
    print(df["sky"].value_counts(dropna=False))
    print()
    print("Rows with missing cLI:")
    print(df[df["cLI"].isnull()])
    print()
    print("Rows with missing weather data:")
    print(df[df["temp"].isnull()])
    print()
    print("Rows with missing windspeed:")
    print(df[df["windspeed"] == -1])


def clean_data(df):
    """
        Takes a DataFrame containing merged game, stadium, and weather data and returns the cleaned DataFrame,
        with converted data types, filled in missing values, encoded columns, dropped unnecessary columns, etc.
    """
    # drop duplicate rows and rows with missing attendance
    df.drop_duplicates(inplace=True)
    df.dropna(subset=["attendance"], inplace=True)

    # split up date column into year, month, day, and day of the week columns
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df.date.dt.month
    df["day"] = df.date.dt.day
    df["day_of_week"] = df.date.dt.dayofweek
    df["day_of_week_name"] = df.date.dt.day_name()

    # covnvert day_or_night column into a binary dummy variable
    df["night_game"] = df["day_or_night"].apply(lambda x: 1 if x == "N" else 0)

    # use record column to create a winning percentage column
    df[['wins', 'losses']] = df['record'].str.split('-', expand=True).astype(int)
    df['win_pct'] = df['wins'] / (df['wins'] + df['losses'])
    df["win_pct"].fillna(0, inplace=True)

    # convert streak column to integer and fill missing values (first games of season) with 0
    df["streak"] = df["streak"].astype(str).str.strip()
    df["streak"].fillna("0", inplace=True)
    df["streak"] = [len(x) if "+" in x else -len(x) if "-" in x else 0 for x in df["streak"]]

    # convert games_behind to float
    df["games_behind"] = df["games_behind"].astype(str).str.strip()
    df["games_behind"] = [0.0 if (x == 'Tied' or x == '0') 
                          else -float(x.replace('up', '').strip()) if 'up' in x
                          else float(x)
                          for x in df["games_behind"]]

    # fill missing cLI values, from what I understand about cLI and the 
    # missing instances I think forward filling is appropriate
    df["cLI"] = df["cLI"].fillna(method="ffill")
    df["cLI"].astype(float)

    # manually fill in the missing temp, precip, windspeed, sky values lost in merge
    # from NYY/NYM doubleheaders played at both stadiums
    df.loc[(df['date'] == '2000-07-08') & (df['team'] == 'NYM'), ['temp', 'sky', 'precip', 'windspeed']] = [76, "sunny", "unknown", 13]
    df.loc[(df['date'] == '2003-06-28') & (df['team'] == 'NYM'), ['temp', 'sky', 'precip', 'windspeed']] = [76, "sunny", "unknown", 6]
    df.loc[(df['date'] == '2008-06-27') & (df['team'] == 'NYM'), ['temp', 'sky', 'precip', 'windspeed']] = [82, "unknown", "unknown", 8]
    df.loc[(df['date'] == '2000-07-08') & (df['team'] == 'NYY'), ['temp', 'sky', 'precip', 'windspeed']] = [77, "sunny", "unknown", 11]
    df.loc[(df['date'] == '2003-06-28') & (df['team'] == 'NYY'), ['temp', 'sky', 'precip', 'windspeed']] = [79, "sunny", "unknown", 7]
    df.loc[(df['date'] == '2008-06-27') & (df['team'] == 'NYY'), ['temp', 'sky', 'precip', 'windspeed']] = [79, "unknown", "unknown", 3]

    # manually replace temp value for rows with temp reported as 0 by mistake:
    # HOU dome is always set a 73, TBR dome is always 72
    df.loc[(df['date'] == '2024-09-08') & (df['team'] == 'HOU'), ["temp"]] = 73
    df.loc[(df['date'] == '2024-06-24') & (df['team'] == 'TBR'), ["temp"]] = 72

    # replace missing windspeeds (had value of -1) with true windspeeds obtained from https://www.wunderground.com/history
    # 0 for the SEA game since it was in a dome
    df.loc[(df['date'] == "2012-04-26") & (df['team'] == 'DET'), ["windspeed"]] = 14
    df.loc[(df['date'] == "2022-05-05") & (df['team'] == 'SEA'), ["windspeed"]] = 0

    # use orig_scheduled column to create a binary dummy variable indicating whether or not the game is a make-up
    df["makeup"] = df["orig_scheduled"].notna().astype(int)

    # convert division_rank, attendance, opening_day, dh, capacity, temp, windspeed to integer
    df["division_rank"] = df["division_rank"].astype(int)
    df["attendance"] = df["attendance"].str.replace(",", "").astype(int)
    df["opening_day"] = df["opening_day"].astype(int)
    df["dh"] = df["dh"].astype(int)
    df["capacity"] = df["capacity"].astype(int)
    df["temp"] = df["temp"].astype(int)
    df["windspeed"] = df["windspeed"].astype(int)

    # drop unnecessary columns
    df.drop(columns=["winning_pitcher", "losing_pitcher", "save", "@", "boxscore", "wins",
                     "losses", "innings", "day_or_night", "number", "record", "w_or_l",
                     "runs_scored", "runs_allowed", "time", "orig_scheduled", "win"], inplace=True)
    
    return df


#TODO: figure out what to do about unknowns in sky and precip
def main():
    """
        Driver for merging, exploring, cleaning, and saving collected data.
    """
    # merge all game, weather, and stadium data
    games_df = merge_data()

    # get some info about the rows in the data, to be addressed in cleaning
    print("PRE-CLEANING")
    print_data_info(games_df)

    # clean the data
    games_df = clean_data(games_df)

    # check the data info after cleaning
    print("POST-CLEANING")
    print_data_info(games_df)

    # save the cleaned data to a CSV file
    games_df.to_csv("data/MLB_games_2000-2024.csv", index=False, encoding="utf-8")
    

if __name__ == "__main__":
    main()
