import argparse
import time
import pandas as pd
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

### Filtered by Type1 and save to filtered_pokemon.csv
    certain_pokemon = df[df["Type1"] == target_type]
    output_path = results_dir / f"filtered_{target_type.lower()}_pokemon.csv"
    certain_pokemon.to_csv(output_path, index=False)

### Summary statistics and save to experiment_log.csv
    end = time.time()
    elapsed_time = end - start
    
    summary_df = pd.DataFrame([
        {
            "type1": target_type,
            "count": certain_pokemon["No"].count(),
            "avg_weight": certain_pokemon["Weight"].mean(),
            "avg_height": certain_pokemon["Height"].mean(),
            "elapsed_time_sec": elapsed_time
        }
    ])

    log_path = results_dir / "experiment_log.csv"
    summary_df.to_csv(log_path,     
                      mode='a', ### append to the file if it already exists, otherwise create a new file.
                      header=not log_path.exists(), ### not true if the file already exists, so we don't write the header again. In our case, header is : type1,count,avg_weight,avg_height
                      index=False,)

    print(f"Filtered data saved to {output_path}")
    print(f"Experiment log saved to {log_path}")
    print(f"걸린 시간: {elapsed_time:.4f}초")

# Only run when this file is executed -> only when I enter "python 03_argparse_example.py" in the terminal
if __name__ == "__main__":
    main()
