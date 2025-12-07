# l101

A modular framework for training, fine-tuning, and evaluating Large Language Models (specifically Pythia) using Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO).

## 📦 Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

```bash
# Clone the repository
git clone https://github.com/woonyee28/l101.git
cd l101

```bash
# Clone the repository
git clone https://github.com/woonyee28/l101.git
cd l101

# Option 1: Using pip (recommended if not using Poetry)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Option 2: Using Poetry
poetry install
```

## 🧩 Modularity & Architecture

The project is designed with modularity at its core, allowing you to easily swap components or reuse them in other scripts. The codebase is organized into the following distinct modules under `src/`:

- **`src/data`**: Handles data ingestion and processing.
  - `dataset_loader.py`: centralized loading of datasets (e.g., BiasDPO) with automatic splitting into train/valid/test sets.
- **`src/models`**: Provides wrappers around HuggingFace models.
  - `pythia_model.py`: Ensures consistent initialization of Pythia models (31m, 70m, 160m) with specific revisions and configurations.
- **`src/training`**: specialized trainers for different alignment techniques.
  - `rlhf_trainer.py`: Implements PPO training loops.
  - `dpo_trainer.py`: Implements DPO training loops.
- **`src/evals`**: Scripts and utilities for evaluating model performance in production-like environments.

## 🔄 How to Reuse

You can import individual modules to build custom training pipelines or analysis scripts.

### 1. Reusing the Data Loader
The `DatasetLoader` class handles downloading, splitting, and preparing datasets automatically.

```python
from src.data.dataset_loader import DatasetLoader

loader = DatasetLoader()
# Automatically downloads and splits the BiasDPO dataset
train_ds, valid_ds, test_ds = loader.load_biasDPO(
    train_ratio=0.7, 
    valid_ratio=0.2, 
    test_ratio=0.1
)

print(f"Train size: {len(train_ds)}")
```

### 2. Loading a Model
Use `PythiaModel` to load supported Pythia variants safely with the correct tokenizer and configuration.

```python
from src.models.pythia_model import PythiaModel

# Load Pythia-70m-deduped
model_wrapper = PythiaModel(name="EleutherAI/pythia-70m-deduped")

# Generate text
output = model_wrapper.generate("The future of AI is", max_len=50)
print(output)
```

### 3. Custom Training Loop
You can initialize the `RLHF_PPO_Trainer` with your own model and dataset instances.

```python
from src.training.rlhf_trainer import RLHF_PPO_Trainer
from src.models.pythia_model import PythiaModel
from src.data.dataset_loader import DatasetLoader
from trl import PPOConfig

# 1. Setup Data
loader = DatasetLoader()
train_ds, valid_ds, _ = loader.load_biasDPO()

# 2. Setup Model
model_wrapper = PythiaModel("EleutherAI/pythia-70m-deduped")

# 3. Configure PPO
config = PPOConfig(output_dir="./my_experiment")

# 4. Initialize Trainer
trainer = RLHF_PPO_Trainer(
    model=model_wrapper.model,
    reward_model_base="EleutherAI/pythia-70m-deduped", # or another model path
    reward_model_config=config, # passing PPOConfig as reward config for example
    value_model=None, # or specific value model
    processing_class=model_wrapper.tokenizer,
    train_dataset=train_ds,
    valid_ds=valid_ds,
    args=config
)

# 5. Train
trainer.train()
```

## 🚀 Running Experiments

Pre-defined configurations are available in the `configs/` directory. You can run training scripts from the `experiments` folder (or `src/evals` scripts) using these configurations.

```bash
# Example: Run 31m RLHF experiment
python src/evals/run_31m.sh
```

## 📜 License
[MIT](LICENSE)
