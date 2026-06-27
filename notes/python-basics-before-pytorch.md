
# Python Basics before PyTorch

This note summarizes basic Python tools that are useful before learning PyTorch.
The goal is not to memorize syntax, but to learn tools for running and recording experiments.

## pathlib

### What is pathlib?

`pathlib` is used to handle file and folder paths in Python.
It is useful when saving experiment results, checkpoints, logs, and plots.

### Why it matters for research code

In experiments, we often need to create folders like:

- `results/`
- `checkpoints/`
- `plots/`
- `logs/`

Using `pathlib` makes this cleaner than writing raw string paths.

### Example

```python
from pathlib import Path

output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

result_file = output_dir / "results.csv"

print(result_file)
