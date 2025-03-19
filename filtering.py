import pandas as pd

# original stadium capacity CSV from https://github.com/tkh5044/, 2017-2024 data collected from https://www.seamheads.com/ballparks/
# retrosheet game data from https://www.retrosheet.org

def filter_stadium_capacity():
   """
      Filters down the staidum capacity data to only include data from 2000 on.
   """
   stadium_cap = pd.read_csv('data/stadium_capacity_full.csv')
   stadium_cap = stadium_cap[stadium_cap["Year"] >= 2000]
   stadium_cap.to_csv('data/stadium_capacity.csv', index=False, encoding='utf-8')


def filter_retrosheet_data():
   """
      Filters down the retrosheet data to only include data from 2000 on.
   """
   retrosheet = pd.read_csv('data/retrosheet_gameinfo.csv')
   retrosheet = retrosheet[retrosheet["season"] >= 2000]
   retrosheet.to_csv('data/retrosheet_gameinfo_filtered.csv', index=False, encoding='utf-8')


def main():
   """
    Filters down stadium capacity and retrosheet data.
   """
   filter_stadium_capacity()
   filter_retrosheet_data()
   

if __name__ == "__main__":
    main()