# Football Match Predictor

It is a simple python program that uses the last 5 years of Premier league data to predict the outcome of a match between 
any two teams of the league.

The model works by first assigning an attack value and defence value to every team in the data set by using Maximum Likelihood Estimation (MLE) 
meaning it gives the best fit rating for each team looking across all 1900 matches. A team's expected goals are calculated by taking their 
attack strength and multiplying it by their opponents defence strength with home advantage factor accounted for. These values are then 
plugged into Poisson distribution which basically estimates the likelihood of random events occurring multiple times in a fixed interval which 
then gives us our win draw loss probabilities. 

We get the data by reading a csv file containing a lot of various info of the last 5 premier league seasons from football-data.co.uk and it 
sorts it by getting rid of all the columns of information that we don't care about. It keeps the date on which the match occurred what 
two teams played how many goals each team scored and who won.

To run, clone the repository, make sure to install the required libraries (pandas, numpy, scipy) using pip, and run Poisson_model.py from the 
src folder. You will then be prompted to enter two Premier League team names.

## Example: 
Enter home team: brentford

Enter away team: arsenal

Brentford win: 23.0%

Draw: 23.3%

Arsenal win: 53.7%

## Results:

Model trained on 2021/22 – 2024/25 seasons and tested on 2025/26 (held-out data the model never saw):

- Accuracy: 45.3%
- Brier score: 0.625 vs baseline 0.642

Better than the baseline by 2.7%. Accuracy is in line with the theoretical ceiling for football prediction (~50-55%), given the high unpredictability of the sport.
