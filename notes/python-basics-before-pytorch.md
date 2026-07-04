
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

### 4. What I learned
I learned that we use time.sleep() to delay for a certain amount of time.
It was surprising that time.time() returns the time passed since January 1, 1970, 00:00:00 (UTC) till now, and the current time is actually obtainable by time.localtime().
I also learned that we can measure the execution time of certain block of codes by subtracting end time by start time, like this:

```python
start = time.time()
...
# codes that I want to measure time
...
end = time.time()
```


### 5. Later use in PyTorch experiments
I can use `time` to measure the exact code execution time. Also, it can be used evaluate the speed of a model, by measuring inference time.

## argparse

### 1. What is it?
`argparse` is a Python module that makes it easy for setting options/arguments when running Python files in the terminal.

### 2. Wht is it useful?
In experiments, it can be frustrating to change the values like epochs and model every time we execute the time.
Therefore, we use `argparse` so that we can change the execution option without actually changing the code.

### 3. Minimal example
I made parse_args() function that takes in Name, Type1, Type2, Height, Weight, Legendary as command line arguments. 
I also have set the default argument and decided if an argument is required or not.

The code is in:
`python_basics/03_argparse_example.py`

### 4. What I learned
I learned that `argparse` is a tool that allows you to adjust settings from the command line terminal that were embedded in the code.

### 5. Later use in PyTorch experiments
In later experiments, I can use `argparse` to change the learning rate, epochs or batch size when performing machine learnging. 
Also, `argparse` adds `--help` that displays parser’s help message.

## pandas

### What is it?
`pandas` is a Python library for working with table-like data.

### Why is it useful?
Research experiments often produce CSV files or json files. I can use pandas to read, filter, summarize, and analyze those results.

### Minimal example
See: `run_pokemon_experiment.py`

### What I learned
I learned how to read a CSV file, select columns, filter rows, and group data.

### Later use in PyTorch experiments
I can use pandas to analyze experiment results such as loss, accuracy, learning rate, batch size, and training time.

## matplotlib

### What is it?
`matplotlib` is a Python library for creating plots and visualizations.

### Why is it useful?
It helps visualize experiment results such as loss curves, accuracy comparisons, and latency measurements.

### Minimal example
See: `run_pokemon_experiment.py`

### What I learned
I learned how to create a simple plot and save it as an image file.

### Later use in PyTorch experiments
I can use matplotlib to plot training loss, validation accuracy, and experiment comparisons.

## numpy

### What is it?
`numpy` is a Python library for numerical computation.

### Why is it useful?
Many ML libraries, including PyTorch and pandas, are connected to array/tensor-style computation.

### Minimal example
Later in PyTorch tensor practice.

### What I learned
I learned that numpy arrays are useful for numerical data, but I do not need to deeply study numpy before starting PyTorch.

### Later use in PyTorch experiments
Understanding numpy arrays will make PyTorch tensors easier to understand.

