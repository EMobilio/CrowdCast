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

    #merged_df.to_csv("data/merged_data.csv", index=False)
    return merged_df

def clean_data(df):
    """
    """
    # use record column to create a winning percentage column
    df[['wins', 'losses']] = df['record'].str.split('-', expand=True).astype(int)
    df['win_pct'] = df['wins'] / (df['wins'] + df['losses'])

    # convert games_behind to float
    df['games_behind'] = df['games_behind'].astype(str).str.strip()
    df["games_behind"] = [0.0 if (x == 'Tied' or x == '0') 
                          else -float(x.replace('up', '').strip()) if 'up' in x
                          else float(x)
                          for x in df["games_behind"]]


    # convert runs_scored, runs_allowed, division_rank, and attendance to integer
    df["runs_scored"] = df["runs_scored"].astyope(int)
    df["runs_allowed"] = df["runs_allowed"].astyope(int)
    df["division_rank"] = df["division_rank"].astype(int)
    df["attendance"] = df["attendance"].str.replace(",", "").astype(int)

    # drop duplicate rows and rows with missing attendance
    df = df.drop_duplcates(inplace=True)
    df = df.dropna(subset=["attendance"], inpace=True)
    
    return df


def main():
    """
    """
    all_data_df = merge_data()

    # get some info about the rows in the data
    print("DF Shape:", all_data_df.shape)
    print("DF Shape after dropping duplicates:", all_data_df.drop_duplicates().shape)
    print("Number of missing attendance values:", all_data_df[all_data_df.attendance.isnull()].shape)
    print("Number of double header games with attendance data:", all_data_df[((all_data_df.attendance.notnull()) & (all_data_df.dh == 1))].shape)

    all_data_df = clean_data(all_data_df)

    

if __name__ == "__main__":
    main()
