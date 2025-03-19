import pandas as pd

# original CSV from https://github.com/tkh5044/, 2017-2024 data collected from https://www.seamheads.com/ballparks/

def main():
   """
    Filters down the staidum capacity data to only include data from 2000 on.
   """
   stadium_cap = pd.read_csv('data/stadium_capacity_full.csv')
   stadium_cap = stadium_cap[stadium_cap["Year"] >= 2000]
   stadium_cap.to_csv('data/stadium_capacity.csv', index=False, encoding='utf-8')

if __name__ == "__main__":
    main()