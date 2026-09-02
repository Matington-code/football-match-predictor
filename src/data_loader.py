import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent

def load_seasons():
    data_folder = project_root / "data"
    csv_files = sorted(data_folder.glob("season*.csv"))


    all_seasons =[]
    for file in csv_files:
        df = pd.read_csv(file)
        all_seasons.append(df)

    combined = pd.concat(all_seasons)
    combined = combined[["Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR"]]

    combined["Date"] = pd.to_datetime(combined["Date"], dayfirst=True)
    combined = combined.sort_values("Date")
    combined = combined.reset_index(drop=True)
    return combined








