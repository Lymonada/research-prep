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
    
    data = {
                "No": [72, 126, 115, 136, 30, 4, 75, 99, 78, 16],
                "Name": [
                    "Tentacool", "Magmar", "Kangaskhan", "Flareon", "Nidorina", 
                    "Charmander", "Graveler", "Kingler", "Rapidash", "Pidgey"
                ],
                "Type1": [
                    "Water", "Fire", "Normal", "Fire", "Poison", 
                    "Fire", "Rock", "Water", "Fire", "Normal"
                ],
                "Type2": [
                    "Poison", None, None, None, None, 
                    None, "Ground", None, None, "Flying"
                ],
                "Height": [0.9, 1.3, 2.2, 0.9, 0.8, 0.6, 1.0, 1.3, 1.7, 0.3],
                "Weight": [45.5, 44.5, 80.0, 25.0, 20.0, 8.5, 105.0, 60.0, 95.0, 1.8],
                "Legendary": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            }

    df = pd.DataFrame(data)
    certain_pokemon = df[df["Type1"] == args.type1]
    end = time.time()

    print(certain_pokemon)
    print(f"개수 : {certain_pokemon['No'].count()}")
    print(f"평균 Weight : {certain_pokemon['Weight'].mean()}")
    print(f"평균 Height : {certain_pokemon['Height'].mean()}")

    print(f"걸린 시간: {end - start:.4f}초")


# Only run when this file is executed -> only when I enter "python 03_argparse_example.py" in the terminal
if __name__ == "__main__":
    main()
