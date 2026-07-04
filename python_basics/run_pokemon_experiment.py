import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "data" / "pokemon.csv"
results_dir = BASE_DIR / "results"
results_dir.mkdir(exist_ok=True)

def parse_args():
    
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--type1", type=str, required=True)

    return parser.parse_args()

def main():
    
    start = time.time()
    args = parse_args()
    df = pd.read_csv(data_path)
    
    target_type = args.type1.capitalize()
    certain_pokemon = df[df["Type1"] == target_type]

    output_path = results_dir / f"filtered_{target_type.lower()}_pokemon.csv"
    certain_pokemon.to_csv(output_path, index=False)

    end = time.time()
    print(certain_pokemon)
    print(f"개수 : {certain_pokemon['No'].count()}")
    print(f"평균 Weight : {certain_pokemon['Weight'].mean()}")
    print(f"평균 Height : {certain_pokemon['Height'].mean()}")
    print(f"걸린 시간: {end - start:.4f}초")

# Only run when this file is executed -> only when I enter "python 03_argparse_example.py" in the terminal
if __name__ == "__main__":
    main()
