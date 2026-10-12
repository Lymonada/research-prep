# research-prep

**Hands-on Python, PyTorch, and machine learning experiments.**

This is my personal study repository as an undergraduate student in Computer Science and Engineering at UNIST. I use it to implement models, run small experiments, and document what I learn while exploring **Efficient AI, ML Systems, and Computer Architecture**.

The projects below are learning exercises—not published research or production-ready systems.

## Featured Projects

### 1. Transformer Implementation from Scratch (PyTorch)

Implemented the core components of an encoder–decoder Transformer using PyTorch modules, without relying on `nn.Transformer`:

- Token embeddings, sinusoidal positional encoding, scaled dot-product and multi-head attention
- Encoder/decoder blocks, feed-forward networks, residual connections, and layer normalization
- Padding and causal attention masks
- Training on a sequence-copy task and autoregressive greedy decoding for variable-length batches

**[Implementation](pytorch_basics/12_Transformer.py) · [Colab notebook](notebooks/Transformer.ipynb) · [Detailed notes](notes/10_Transformer_Implementation_Final_Notes.md)**

### 2. Mini Retrieval-Augmented Generation (RAG) Pipeline

Built a small end-to-end RAG pipeline over Markdown study notes:

```text
Markdown files -> overlapping word chunks + source metadata
               -> SentenceTransformer embeddings
               -> cosine similarity + top-k retrieval
               -> context-augmented prompt -> LLM via OpenRouter
```

- Implemented cosine similarity and retrieval with PyTorch tensor operations.
- Used `all-MiniLM-L6-v2` to embed document chunks and questions.
- Ran a simple with/without-context test to check whether the generated answer depended on retrieved information.

**[Implementation](pytorch_basics/13_mini_RAG.py) · [Colab notebook](notebooks/mini_RAG.ipynb) · [Retrieval notes](notes/11_Mini%20RAG%20Retrieval%20Pipeline.md) · [Generation notes](notes/12_Mini%20RAG%20Augmentation%20and%20Generation.md)**

### 3. CNN Architecture & Transfer Learning Experiments (CIFAR-10)

Trained basic and deeper CNNs, then compared two transfer-learning setups using ImageNet-pretrained ResNet18.

| Model | Test accuracy |
| --- | ---: |
| Basic CNN | 74.66% |
| Deeper CNN | 80.90% |
| ResNet18 (frozen backbone; classifier trained) | 78.09% |
| ResNet18 (final residual block + classifier fine-tuned) | 89.54% |

These are results from my learning experiments, **not a controlled benchmark** between architectures: the pretrained models differ from the CNNs in initialization and preprocessing.

**[Results CSV](results/CIFAR10/run_summary.csv) · [CNN comparison notes](notes/04_CNN_CIFAR10_depth_comparison.md) · [ResNet18 notes](notes/05_ResNet18_CIFAR10_transfer_learning_comparison.md)**

Code: [Basic CNN](pytorch_basics/06_CNN_CIFAR10.py) · [Deeper CNN](pytorch_basics/07_DeepCNN_CIFAR10.py) · [Frozen ResNet18](pytorch_basics/08_ResNet18_Frozen_CIFAR10.py) · [Fine-tuned ResNet18](pytorch_basics/09_ResNet18_FineTuning_CIFAR10.py)

## Other Learning Exercises

- **ML fundamentals:** linear/logistic regression, MLP and CNN classification on MNIST
- **Sequence models:** LSTM and GRU implementations and a small stock-price prediction comparison ([notes](notes/06_LSTM_GRU_stock_prediction_comparison.md))
- **Python tools:** experiments using `pathlib`, `argparse`, CSV logging, plotting, and related utilities

## Repository Structure

```text
research-prep/
├── python_basics/   # Python utilities and experimentation practice
├── pytorch_basics/  # Model implementations and training scripts
├── notebooks/       # Original-style Colab learning and experiment records
├── notes/           # Concept explanations and experiment write-ups
├── results/         # Saved experiment summaries
├── plots/           # Training and prediction visualizations
└── data/            # Small example datasets
```

## Notes on Running the Code

- The repository consists of **standalone learning scripts and Colab notebooks**, not a single installable package. The notebooks preserve step-by-step learning tests and comments; the accompanying notes explain the experiments and their assumptions.
- Dependencies vary by example and include packages such as `torch`, `torchvision`, `sentence-transformers`, `openai`, `pandas`, and `matplotlib`.
- The Mini RAG script was developed for **Google Colab** and loads an `OPENROUTER_API_KEY` from Colab Secrets. Running it requires external model downloads and API access. Do not commit API keys.

**Current study focus:** computer architecture and systems programming, with the longer-term goal of understanding how ML models run efficiently on real hardware.
