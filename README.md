# temporary - `stlenc`

> An attempt at distilling kernel into a Transformer encoder architecture.

This repository contains the experimental code, benchmarks, and training scripts for distilling kernel-based methods into a Transformer encoder model. 

## 📂 Project Structure

* **`src/`**: Core source code and model architectures.
* **`train-src/`**: Scripts and utilities for training the models.
* **`tests-src/`**: Unit tests and validation scripts.
* **`datapreproc/`**: Data preprocessing pipelines and utilities.
* **`efficient-benchmarks/`**: Benchmarking scripts to evaluate model efficiency and performance.
* **`results_speedup/`**: Stored results, graphs, and metrics regarding speedup and performance.
* **`slurm_outputs/`**: Log files and outputs from SLURM cluster jobs.
* **`wip/`**: Work-in-progress scripts (to be cleaned) and experimental notebooks.
* **`utils.py`**: Shared helper functions used across the project.
* **`requirements.txt`**: Python dependencies required to run the project.

## 🚀 Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/gaoithee/stlenc.git](https://github.com/gaoithee/stlenc.git)
   cd stlenc

```

2. **Create a virtual environment (optional but recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

```


3. **Install dependencies:**
```bash
pip install -r requirements.txt

```



## 💻 Usage

*(Note: Update this section with specific commands once the entry points are finalized)*

**Data Preprocessing:**

```bash
# Example command for data preprocessing
# python datapreproc/preprocess.py --config config.yaml

```

**Training:**

```bash
# Example command to kick off training
# python train-src/train.py --batch_size 32

```

**Benchmarking:**

```bash
# Example command to run efficiency benchmarks
# python efficient-benchmarks/run_benchmark.py

```

## 🧪 Testing

To run the test suite, navigate to the `tests-src` directory and run:

```bash
pytest tests-src/

```
