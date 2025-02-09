# CrowdCast

## Project Description

Baseball has been considered America's pastime for nearly two centuries, and spending a day at the ballpark has been a favorite activity of Americans throughout that time. In this project, I propose developing a predictive model to estimate attendance at future MLB games and determine which factors are most significant in bringing people to the ballpark. By analyzing historical game data and factors like weather conditions, team performance, and schedule factors, I will build a model that can be used to understand how to get more people attending games.

## Goals

The goal of this project is to develop a model that can successfully predict the number of people in attendance at an MLB game based on factors including weather conditions, stadium capacity, day of the week, team position in standings, recent team perofmance, and championship importance, among others. I hope to develop multiple models that will improve on one another and to identify which features have the greatest impact on game attendance.

## Data Collection

For this project, most of the data will come from scraping [Baseball Reference](https://www.baseball-reference.com/) schedules and game logs. I will likely gather data on all MLB games over the last 5 (or perhaps 10) years, but I think it would be best to exclude the 2020 and 2021 seasons due to attendance restrictions caused by COVID. Features that I will gather will include:
### Game/Schedule Factors:
- Attendance: the number of tickets sold for a game
- Stadium capacity: the number of fans a stadium can hold
- Day/night: whether the game was played during the day or at night
- Start time: the time of day at which the game started
- Day of the week: on which day of the week the game was played
### Team Performance:
- Games behind: how may games behind (or ahead) a team is of their division leader in the standings
- Streak: how many games in a row a team has won or lost going into a game
- Winning percentage: a team's winning percentage going into a game
- Rank: a team's ranking in their division
- Runs per game: the average number of runs scored per game by a team in a season up to a game
- Runs allowed per game: the average number of runs given up per game by a team in a season up to a game
- Last 10 winning percentage- a team's winning percentage over their previous 10 games leading up to a game
- Championship leverage index: how important a game is to a team's probability of winning the World Series
### Weather Conditions:
- Temperature: the temperature at the start of the game measured in degrees fahrenheit
- Wind Speed: the wind speed during the game measured in mph
- Precipitation: whether or not there was any precipitation during the game
- Dome/no dome- whether or not the game was played under a dome
- Weather description: categorical description of the weather (sunny, cloudy, overcast, etc.)

Some of these features, like last 10 winning percentage and runs per game, will have to be extracted. Other data sources like a weather API and [Retrosheet](https://www.retrosheet.org/) may ultimately be necessary for collecting more specific weather/game data. I would also potentially include factors like promotional giveaways, average player age, and presence of star players if I can find sources for this data and viable ways of extracting these features.

## Data Modeling

I plan to start by trying to fit a linear model to establish a baseline for prediction. Then I think it would be appropriate to use a random forest or XGBoost to try to capture non-linear relationships in the data and give better predictions.

## Data Visualization

I will use heatmaps to show correlation between attendance and external factors. I will utilize feature importance graphs to identify which factors are most significant in influencing attendance. I will also use regression plots to show relationships between individual features and attendance.

## Testing

I will split the collected game data into sets for training and testing, most probably 80% for training and 20% for testing, though I am not quite sure whether it would be best to split the data completely randomly or to split it by season. Once the 2025 MLB season starts, I also plan on using the early season game data from April to test my model. I will likely use metrics like RMSE and $$R^2$$ to evaluate the model's accuracy.
