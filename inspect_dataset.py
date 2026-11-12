import os
import pandas as pd

def inspect():
    path = "data/processed/water_demand_clean.csv"
    if not os.path.exists(path):
        from prepare_dataset import prepare_data
        prepare_data()
    df = pd.read_csv(path)
    print("=" * 50)
    print("DATASET QUALITY & SUMMARY")
    print("=" * 50)
    print(f"Total Usable Records: {len(df)}")
    print("\nDescriptive Statistics:\n", df.describe())

if __name__ == '__main__':
    inspect()
