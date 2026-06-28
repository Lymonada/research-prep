
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

## time

### 1. What is it?
`time` is a Python module for handling time related features. 

### 2. Why is it useful?
In experiments, we can use `time` module to achieve these things:

- measure time
- time conversion
- better formatting of time
- intentional delay between codes

### 3. Minimal example
I measured the time elapsed for reading csv file(pokemon.csv).

The code is in:
`python_basics/02_time_example.py`

### 5. Later use in PyTorch experiments
I can use `time` to measure the exact code execution time. Also, it can be used evaluate the speed of a model, by measuring inference time.

## argparse

### 1. What is it?
`argparse` is a Python module that makes it easy for setting options/arguments when running Python files in the terminal.

### 2. Wht is it useful?
In experiments, it can be frustrating to change the values like epochs and model every time we execute the time.
Therefore, we use `argparse` so that we can change the execution option without actually changing the code.

### 3. Minimal example

