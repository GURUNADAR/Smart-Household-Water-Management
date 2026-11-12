import os
import pandas as pd

def prepare_data(raw_path="data/raw/Pune_HHLD_Survey_Data_v2-0-0.csv", 
                 output_path="data/processed/water_demand_clean.csv"):
    if not os.path.exists(raw_path):
        if os.path.exists("Pune_HHLD_Survey_Data_v2-0-0.csv"):
            raw_path = "Pune_HHLD_Survey_Data_v2-0-0.csv"
        else:
            raise FileNotFoundError(f"Raw dataset not found at {raw_path}")

    print(f"Reading dataset from {raw_path}...")
    df = pd.read_csv(raw_path, encoding='latin1', low_memory=False)

    bill_mask = (
        df['waterbill_consumption_1'].notna() & 
        df['billing_period'].notna() & 
        (df['billing_period'] > 0) & 
        (df['waterbill_consumption_1'] > 0)
    )
    df_clean = df[bill_mask].copy()
    df_clean['daily_litres'] = (df_clean['waterbill_consumption_1'] * 1000.0) / df_clean['billing_period']

    df_clean = df_clean[
        (df_clean['household'] >= 1) & 
        (df_clean['household'] <= 15) & 
        (df_clean['daily_litres'] >= 50) & 
        (df_clean['daily_litres'] <= 3500)
    ].copy()

    dwelling_map = {1: 'Independent House', 2: 'Apartment', 3: 'Row House', 4: 'Other'}
    df_clean['dwelling_type'] = df_clean['dwelling'].map(dwelling_map).fillna('Independent House')
    df_clean['washing_machine'] = df_clean['appliance_own__1'].fillna(0).astype(int)
    df_clean['garden'] = df_clean['garden'].fillna(0).astype(int)
    df_clean['water_source'] = df_clean['tanker_use'].map({1: 'Tanker Supplemental', 0: 'Municipal Piped'}).fillna('Municipal Piped')
    df_clean['rooms'] = df_clean['rooms'].fillna(df_clean['rooms'].median()).astype(float)
    df_clean['household_size'] = df_clean['household'].astype(int)

    selected_features = [
        'household_size', 'dwelling_type', 'rooms', 
        'washing_machine', 'garden', 'water_source', 'daily_litres'
    ]
    df_final = df_clean[selected_features].reset_index(drop=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f"[✓] Processed dataset saved with {len(df_final)} records.")
    return df_final

if __name__ == '__main__':
    prepare_data()
