
# Python Basics before PyTorch

This note summarizes basic Python tools that are useful before learning PyTorch.
The goal is not to memorize syntax, but to learn tools for running and recording experiments.

## pathlib

### 1. What is it?
`pathlib` is a Python module for handling file and folder paths.

### 2. Why is it useful?
In experiments, we often need to create folders like:

- `results/`
- `checkpoints/`
- `plots/`
- `logs/`

Using `pathlib` makes this cleaner than writing raw string paths.

### 3. Minimal example
I created `results/` and `plots/` folders using `Path.mkdir()`.

The code is in:
`python_basics/01_pathlib_example.py`

### 4. What I learned
I learned how to create folders and build file paths using `Path`.  I also learned that mkdir() can have two options: parents and exist_ok.  

When parents = True, it creates parent folder with the folder I created.  
When exist_ok = True, it handles the case when the folder already exists.


### 5. Later use in PyTorch experiments
I can use `pathlib` to save model checkpoints, result CSV files, and loss plots.
