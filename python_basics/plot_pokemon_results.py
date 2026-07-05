import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
plots_dir = BASE_DIR / "plots"
plots_dir.mkdir(exist_ok=True)
plot_path = plots_dir / "type_count.png"

results_dir = BASE_DIR / "results"
log_path = results_dir / "experiment_log.csv"

log_df = pd.read_csv(log_path)
plot_df = log_df.groupby("type1", as_index=False)["count"].last() # To handle cases when there are multiple results of same type1
plt.bar(plot_df["type1"], plot_df["count"])

plt.bar(plot_df["type1"], plot_df["count"])
plt.xlabel("Type1")
plt.ylabel("Count")
plt.title("Pokemon Count by Type1")
plt.tight_layout()
plt.savefig(plot_path)
