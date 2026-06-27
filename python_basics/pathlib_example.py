from pathlib import Path

data_path = Path("data") / "pokemon.txt"
results_dir = Path("results")
plots_dir = Path("plots")

results_dir.mkdir(exist_ok=True)
plots_dir.mkdir(exist_ok=True)

print(data_path)
print(data_path.exists())
