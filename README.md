# PLKT: Probabilistic Logical Knowledge Tracing

PLKT is an interpretable Knowledge Tracing (KT) framework that reframes student performance prediction as a goal-conditioned evidence reasoning process over historical learning behaviors. Unlike conventional "black-box" sequence models, PLKT provides transparent reasoning paths, revealing how and why specific past interactions influence each prediction.

---

## 🚀 Features

- ⚙️ **Beta Embedding**: Instead of representing knowledge states as deterministic vector embeddings, PLKT employs robust Beta-distributed probabilistic embeddings to represent student knowledge states.
- 🧠 **Multi-level Pattern Extraction**: We propose a multi-level pattern extraction mechanism to explicitly capture learning behaviors at different temporal granularities.
---

## 📦 Repository Structure


---

## 🔧 Installation

### Requirements

- Python >= 3.8
- PyTorch >= 1.12
- numpy, pandas, scikit-learn, tqdm

## 🚀 Usage

┌── configs 
│	├── config.py #The config
│	├── data_config.json # Dataset parameters
│	├── kt_config.json # Configuration of relevant model parameters
|
├── pykt
│	├── datasts # Data preprocessing code
|  ├── atdkt_dataloader.py
|  ├── data_loader.py
|  ├── ....
|
├── models # Model code
|  ├── akt.py
|  ├── atkt.py
|  ├── ....
|
├── wandb_ptkt_train.py
|
└── wandb_train.py # Training

---
