import pandas as pd
from pathlib import Path
from src.db.connection import get_engine

RAW_DATA_PATH = Path("Data/raw/diabetic_data.csv")  

def load_raw_data():
    print(f"Reading raw data from {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    engine = get_engine()

    print("Writing to Postgres table 'raw_encounters'...")
    df.to_sql(
        "raw_encounters",
        engine,
        if_exists="replace",   # safe to re-run — drops and recreates the table each time
        index=False
    )
    print("Done. Raw data now lives in Postgres.")

if __name__ == "__main__":
    load_raw_data()