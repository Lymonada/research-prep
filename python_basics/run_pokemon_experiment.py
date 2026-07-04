import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "data" / "pokemon.csv"

def parse_args():
    
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--type1", type=str, required=True)

    return parser.parse_args()

def main():
    
    start = time.time()
    args = parse_args()
    df = pd.read_csv(data_path)
    
    certain_pokemon = df[df["Type1"] == args.type1]
    
    print(certain_pokemon)
    print(f"개수 : {certain_pokemon['No'].count()}")
    print(f"평균 Weight : {certain_pokemon['Weight'].mean()}")
    print(f"평균 Height : {certain_pokemon['Height'].mean()}")

    end = time.time()

    print(f"걸린 시간: {end - start:.4f}초")
    

# Only run when this file is executed -> only when I enter "python 03_argparse_example.py" in the terminal
if __name__ == "__main__":
    main()
