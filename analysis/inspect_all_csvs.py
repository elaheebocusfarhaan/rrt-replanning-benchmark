import os
DATA_DIR = os.environ.get("PHYSICAL_DATA_DIR", "./results/physical")
import pandas as pd

files = [
    os.path.join(DATA_DIR, "rrt.csv"),
    os.path.join(DATA_DIR, "rrt_connect.csv"),
    os.path.join(DATA_DIR, "rrtstar.csv"),
    os.path.join(DATA_DIR, "rrtx.csv")
]

for file in files:
    print("\n" + "="*60)
    print(f"FILE: {file}")
    print("="*60)

    try:
        df = pd.read_csv(file)

        print("\nColumns:")
        print(list(df.columns))

        print("\nHead (first 5 rows):")
        print(df.head())

        print("\nData types:")
        print(df.dtypes)

        print("\nSummary stats:")
        print(df.describe())

        print(f"\nTotal rows: {len(df)}")

    except Exception as e:
        print(f"Error reading {file}: {e}")