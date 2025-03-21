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
       Adds weather data from retrosheet CSV to game_data CSV records, saving the merged data as a new CSV file.
    """
    retrosheet_df = pd.read_csv("data/retrosheet_gameinfo_2000-2024.csv")
    game_data_df = pd.read_csv("data/game_data.csv")

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

    merged_df.to_csv("data/game_data.csv", index=False)
    return merged_df


def main():
    """
    """
    merged_df = merge_data()
    

if __name__ == "__main__":
    main()
