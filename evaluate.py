from Poisson_model import predict_match, teams
from data_loader import load_seasons
import pandas as pd

test_matches = load_seasons(before=None)
test_matches = test_matches[test_matches["Date"] >= "2025-08-01"]

correct = 0
total = 0

for row in test_matches.itertuples():
    if row.HomeTeam not in teams or row.AwayTeam not in teams:
        continue
    h_win, a_win, draw = predict_match(row.HomeTeam, row.AwayTeam)

    # what did our model predict as most likely?
    if h_win > a_win and h_win > draw:
        prediction = "H"
    elif a_win > h_win and a_win > draw:
        prediction = "A"
    else:
        prediction = "D"

    # did it match reality?
    if prediction == row.FTR:
        correct += 1
    total += 1
accuracy = correct / total
print(f"Accuracy: {accuracy:.1%}")

brier = 0
for row in test_matches.itertuples():
    if row.HomeTeam not in teams or row.AwayTeam not in teams:
        continue
    h_win, a_win, draw = predict_match(row.HomeTeam, row.AwayTeam)

    actual_h = 1 if row.FTR == "H" else 0
    actual_d = 1 if row.FTR == "D" else 0
    actual_a = 1 if row.FTR == "A" else 0

    brier += (h_win - actual_h) ** 2 + (draw - actual_d) ** 2 + (a_win - actual_a) ** 2

brier = brier / total
print(f"Brier score: {brier:.4f}")

baseline_h = 0.46
baseline_d = 0.26
baseline_a = 0.28
baseline_brier = ((baseline_h - 1)**2 + baseline_d**2 + baseline_a**2) * 0.46
baseline_brier += (baseline_h**2 + (baseline_d - 1)**2 + baseline_a**2) * 0.26
baseline_brier += (baseline_h**2 + baseline_d**2 + (baseline_a - 1)**2) * 0.28
print(f"Baseline Brier score: {baseline_brier:.4f}")