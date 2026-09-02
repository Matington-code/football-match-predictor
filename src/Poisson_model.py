import numpy as np                   #maths on arrays of numbers
import pandas as pd
from pathlib import Path
from scipy.stats import poisson      # Poisson distribution calculates probabilities given λ
from scipy.optimize import minimize  # optimizer that finds the best α and β values via MLE
from data_loader import load_seasons

project_root = Path(__file__).parent.parent
matches = load_seasons()
teams = sorted(matches["HomeTeam"].unique())

team_index = {}
pos = 0
for team in teams:
    team_index[team]= pos
    pos +=1

def calculate_lambda(params,home_team,away_team):
    h_i = team_index[home_team]
    a_i = team_index[away_team]

    h_attack = params[h_i]
    h_defence = params[h_i + len(teams)]
    h_advantage = params[-1]

    a_attack = params[a_i]
    a_defence = params[a_i + len(teams)]

    h_λ = h_attack * a_defence * h_advantage
    a_λ = a_attack  * h_defence

    return h_λ,a_λ

def neg_log_likelihood(params):
    total = 0
    for row in matches.itertuples():
        h_λ, a_λ = calculate_lambda(params, row.HomeTeam, row.AwayTeam)
        p_home = poisson.pmf(row.FTHG, h_λ)
        p_away = poisson.pmf(row.FTAG, a_λ)
        total += np.log(p_home) + np.log(p_away)
    return -total

def fit_model():
    n = len(teams)
    initial_params = np.ones(n * 2 + 1)
    result = minimize(neg_log_likelihood, initial_params, method="Nelder-Mead", options={"maxiter": 50000})
    return result


def predict_match(home_team, away_team):
    h_λ, a_λ = calculate_lambda(params,home_team,away_team)
    h_win_total = 0
    a_win_total = 0
    draw = 0
    for home_goals in range(11):
        for away_goals in range(11):
            prob = poisson.pmf(home_goals, h_λ) * poisson.pmf(away_goals, a_λ)
            if home_goals > away_goals :
                h_win_total += prob
            elif home_goals < away_goals :
                a_win_total += prob
            else :
                draw += prob

    return h_win_total,a_win_total,draw


params_path = project_root / "params.npy"
if params_path.exists() :
    params = np.load(params_path)
else:
    result = fit_model()
    params = result.x
    np.save(params_path, params)


home_team = input("Enter home team: ").strip().title()
away_team = input("Enter away team: ").strip().title()
if home_team not in teams or away_team not in teams:
    print("Team not found. Check the spelling.")
else:
    h_win, a_win, draw = predict_match(home_team, away_team)
    print(f"{home_team} win: {h_win:.1%}")
    print(f"Draw: {draw:.1%}")
    print(f"{away_team} win: {a_win:.1%}")




