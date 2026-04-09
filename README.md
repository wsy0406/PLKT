# PLKT: Probabilistic Logical Knowledge Tracing

PLKT is an interpretable Knowledge Tracing (KT) framework that reframes student performance prediction as a goal-conditioned evidence reasoning process over historical learning behaviors. Unlike conventional "black-box" sequence models, PLKT provides transparent reasoning paths, revealing how and why specific past interactions influence each prediction.

---

## 🚀 Features

- ⚙️ **Beta Embedding**: Instead of representing knowledge states as deterministic vector embeddings, PLKT employs robust Beta-distributed probabilistic embeddings to represent student knowledge states.
- 🧠 **Multi-level Pattern Extraction**: We propose a multi-level pattern extraction mechanism to explicitly capture learning behaviors at different temporal granularities.
---


## 🔧 Installation

### Requirements

- Python >= 3.8
- PyTorch >= 1.12
- numpy, pandas, scikit-learn, tqdm

## 📦 Repository Structure

```text
configs
├── config.py              # The config
├── data_config.json       # Dataset parameters
└── kt_config.json         # Configuration of relevant model parameters

pykt
├── datasts                # Data preprocessing code
│   ├── atdkt_dataloader.py
│   ├── data_loader.py
│   └── ...
├── models                 # Model code
│   ├── akt.py
│   ├── atkt.py
│   └── ...
├── wandb_plkt_train.py
└── wandb_train.py         # Training
```

---

## Reproducing
### Run wandb_plkt_train.py
Having installed all necessary packages, you can run wandb_plt_train.py using
```text
python wandb_plkt_train.py --dataset_name assist2009 --device cuda:0 --pattern_type multi_level_bias --emb_type beta --pattern_level 5 --bias_weight 0.7 --learning_rate 0.0001 --seed 42 --dropout 0.2
```

The table above summarizes the key parameters for running the model. To train on a specific dataset, set `dataset_name` accordingly. You can use `pattern_type` to choose between multi-level and single-level PLKT, and `pattern_level` to control the level at which patterns are extracted. Other hyperparameters, such as `lr`, `seed`, and `dropout`, can be adjusted as needed for different experimental settings.
