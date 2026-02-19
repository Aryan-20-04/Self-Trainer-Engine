<div align="center">

# 🧠 SelfTrainerEngine

### _Lifecycle-Aware, Tunable, Experiment-Tracked, Drift-Detecting ML Orchestration Framework_

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Features](#-features) • [Installation](#️-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Roadmap](#-roadmap)

</div>

---

## 📖 Overview

**SelfTrainerEngine** is a production-ready, automated machine learning framework designed for tabular data. It intelligently handles the entire ML lifecycle—from task detection to model deployment—with built-in hyperparameter tuning, drift detection, experiment tracking, and full reproducibility.

Designed for production-grade tabular ML systems requiring automation, monitoring, explainability, and reproducibility.

### Why SelfTrainerEngine?

- 🎯 **Zero-config ML**: Automatically detects task type, handles imbalance, and selects optimal models
- ⚙️ **Config-Driven Engine**: Switch between `dev` and `full` modes for fast experimentation or deep training
- 🔧 **Intelligent Hyperparameter Tuning**: Optuna-powered tuning applied to top-2 models only — no wasted compute
- 🔍 **Built-in Explainability**: SHAP integration for global and instance-level interpretability
- ⚖️ **Smart Imbalance Handling**: Automatic detection and correction for class imbalance
- 📊 **Proper Validation**: Clean train/validation/test splits with threshold optimization
- 📈 **Drift Detection**: PSI-based feature-level drift monitoring with persisted baselines
- 🧪 **Experiment Tracking**: Structured logging of every run with full metadata persistence
- 🚀 **Production-Ready**: Designed with best practices for real-world deployment

---

## ✨ Features

### 🤖 Automatic Task Detection

Intelligently identifies your ML problem:

- **Binary/Multi-class Classification**
- **Regression**

### ⚙️ Config-Driven Engine (dev / full mode)

Control engine behavior via `EngineConfig`:

```python
from core.engine import SelfTrainerEngine, EngineConfig

# Fast experimentation
config = EngineConfig(mode="dev")

# Deep production training
config = EngineConfig(mode="full")

engine = SelfTrainerEngine(config=config)
```

`EngineConfig` controls:

- CV folds
- SHAP sample size
- Tree estimators
- Drift threshold
- Optuna timeout / trials

### 🔧 Hyperparameter Optimization (Optuna)

The engine evaluates all candidate models, selects the **top 2**, and tunes only those using Optuna — preventing unnecessary compute waste.

- Applied to: XGBoost, RandomForest, LightGBM, LogisticRegression
- Uses cross-validation scoring with the correct task metric
- Best params are automatically applied and saved in metadata
- Timeout-based tuning for production-realistic behavior

### ⚖️ Imbalance Handling

Automatically detects and corrects class imbalance using:

- `class_weight="balanced"` for scikit-learn models
- `scale_pos_weight` for XGBoost
- Built-in class balancing for LightGBM

### 📈 Smart Metric Selection

Chooses the right metric based on your data:

- **ROC-AUC** for imbalanced classification
- **F1-weighted** for balanced classification
- **RMSE** for regression tasks

### 🎯 Threshold Optimization

_(Classification only)_

Learns optimal probability thresholds using the validation set for maximum F1-score.

### 🔬 Model Selection

Automatically evaluates multiple model families:

- **Random Forest**
- **Gradient Boosting**
- **XGBoost**
- **LightGBM**
- **Logistic/Linear Regression**

### 🧩 Proper Data Splitting

Ensures no data leakage with clean splits:

- **Train** (60%): Model learning
- **Validation** (20%): Threshold tuning & model selection
- **Test** (20%): Final evaluation

### 💡 Explainability

SHAP-powered interpretability (stabilized with config-driven sample sizes):

- **Global explanations**: Feature importance across entire dataset
- **Instance-level explanations**: Why a specific prediction was made
- Automatic TreeExplainer/KernelExplainer fallback
- Works correctly after model reload

### 📊 Structured Experiment Logging

Every training run is automatically logged under `experiments/run_YYYYMMDD_HHMMSS/` with:

- CV results
- Best model & best params
- Test score
- Threshold
- Dataset size
- Timestamp

### 📦 Full Metadata Persistence

Model metadata now includes: `best_params`, `baseline` (for drift), `threshold`, `cv_results`, and `test_score` — making every model fully reproducible.

### 🌊 PSI-Based Drift Detection

- Baseline is built from training data and stored in metadata
- Reloadable after model load
- PSI calculated per feature
- Drifted features reported individually

---

## 🔄 Engine Flow

```
fit()
 ├── Detect task
 ├── Split (train/val/test)
 ├── Build drift baseline
 ├── Evaluate all models
 ├── Select top 2
 ├── Tune with Optuna
 ├── Select best tuned model
 ├── Threshold optimization
 ├── Final test evaluation
 ├── Save model + metadata
 ├── Log experiment
 └── Initialize SHAP explainer
```

---

## 📊 Visualizations

Generate professional documentation images from your actual model results:

```bash
python generate_images.py
```

This creates visualizations in `docs/images/`:

- **Architecture diagram** - Pipeline overview
- **Workflow diagram** - Step-by-step process with your actual metrics
- **Performance comparison** - Your actual model scores
- **SHAP feature importance** - Top 5 features from your model
- **Metrics summary** - Complete dashboard of your results

All charts use your real training data, not dummy values!

---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/SelfTrainerEngine.git
cd SelfTrainerEngine
```

2. **Create a virtual environment** _(recommended)_

```bash
# macOS/Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

### Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
shap>=0.41.0
matplotlib>=3.4.0
optuna>=3.0.0
```

---

## 🚀 Quick Start

### Basic Usage

```python
import pandas as pd
from core.engine import SelfTrainerEngine

df = pd.read_csv("data/your_dataset.csv")

engine = SelfTrainerEngine()
engine.fit(df, target="target_column")

engine.summary()

engine.explain_global()

sample = df.drop("target_column", axis=1).sample(1)
engine.explain_instance(sample)
```

### Config-Driven Usage

```python
from core.engine import SelfTrainerEngine, EngineConfig

# Dev mode: fast runs, fewer folds, fewer estimators
config = EngineConfig(mode="dev")
engine = SelfTrainerEngine(config=config)
engine.fit(df, target="Class")

# Full mode: deep training, more folds, longer Optuna timeout
config = EngineConfig(mode="full")
engine = SelfTrainerEngine(config=config)
engine.fit(df, target="Class")
```

### Classification Example

```python
# Credit card fraud detection
df = pd.read_csv("data/creditcard.csv")

engine = SelfTrainerEngine()
engine.fit(df, target="Class")

# Automatically handles:
# - Imbalanced classes
# - Optimal threshold selection
# - Model selection (ROC-AUC metric)
# - Hyperparameter tuning (Optuna)

engine.summary()
# Output:
# Task: classification
# Best Model: XGBoostClassifier
# Test ROC-AUC: 0.9847
# Optimal Threshold: 0.23
```

### Regression Example

```python
# House price prediction
df = pd.read_csv("data/housing.csv")

engine = SelfTrainerEngine()
engine.fit(df, target="price")

# Automatically uses RMSE for evaluation
engine.summary()
# Output:
# Task: regression
# Best Model: LightGBMRegressor
# Test RMSE: 23456.78
```

### Full Lifecycle Example (Training → Monitoring → Reload)

```python
engine = SelfTrainerEngine()
engine.fit(df, target="Class")

# Predict
preds = engine.predict(df.drop("Class", axis=1))

# Drift Detection
engine.check_drift(df.drop("Class", axis=1))

# Reload Model
new_engine = SelfTrainerEngine()
new_engine.load(engine.model_path, engine.meta_path)

# Predict after reload
prediction = new_engine.predict(df.drop("Class", axis=1).sample(1))
```

---

## 📂 Project Structure

```
SelfTrainerEngine/
│
├── core/                      # Core ML engine components
│   ├── engine.py             # Main orchestrator + EngineConfig
│   ├── evaluator.py          # Performance evaluation
│   ├── model_registry.py     # Model configurations (config-driven estimators)
│   ├── task_detector.py      # Task type detection
│   ├── threshold.py          # Threshold optimization
│   └── trainer.py            # Model training logic
│
├── optimization/             # Hyperparameter tuning
│   └── optuna_tuner.py       # Optuna integration
│
├── explainability/           # Explainability module
│   └── explainer.py          # SHAP integration
│
├── monitoring/               # Drift detection & experiment tracking
│   ├── baseline.py           # Baseline builder
│   ├── drift_detector.py     # PSI-based drift detector
│   └── experiment_tracker.py # Experiment logging
│
├── versioning/               # Model versioning
│   └── model_store.py        # Save/load model artifacts
│
├── experiments/              # Experiment logs
│   ├── registry.json
│   └── run_YYYYMMDD_HHMMSS/
│       ├── metadata.json
│       └── metrics.json
│
├── models/                   # Saved model artifacts
│
├── reports/                  # Generated SHAP plots and reports
│
├── tests/
│   ├── test_self_trainer.py  # Core functionality tests
│   └── test_stress.py        # Edge case & stress tests (122 tests)
│
├── .gitignore
├── conftest.py
├── main.py                   # Example usage script
├── requirements.txt
└── README.md
```

---

## 📚 Documentation

### Customizing Visualizations

To update the visualizations with your own model results, edit `generate_images.py`:

```python
# Example Output
model_scores = {
    'LogisticRegression': 0.972667,
    'RandomForest': 0.961487,
    'XGBoost': 0.990387,      # Your best score
    'LightGBM': 0.977915,
    'MLP': 0.729110
}

# Example top features
top_features = [
    ('V10', -3.81787),
    ('V14', -2.05648),
    ('V12', -1.62463),
    ('V4', -1.18593),
    ('V16', -0.73025)
]

# Example performance metrics
best_model = 'XGBoost'
optimal_threshold = 0.9856
validation_f1 = 0.8235
test_score = 0.977165
```

Then regenerate:

```bash
python generate_images.py
```

### API Reference

#### `SelfTrainerEngine`

**Methods:**

- **`fit(df: pd.DataFrame, target: str, test_size: float = 0.2, val_size: float = 0.2)`**
  - Trains the engine on your dataset
  - Automatically detects task, handles imbalance, tunes and selects model
  - **Parameters:**
    - `df`: Input DataFrame containing features and target
    - `target`: Name of the target column
    - `test_size`: Fraction of data held out for final testing (default 0.2)
    - `val_size`: Fraction of data held out for validation and threshold tuning (default 0.2)
- **`summary()`**
  - Prints comprehensive performance metrics
  - Shows optimal threshold (if classification)
  - Displays best model, best params, and evaluation scores
- **`explain_global(save_path: str = None)`**
  - Generates SHAP summary plot
  - Shows feature importance across dataset
  - **Parameters:**
    - `save_path`: Optional path to save the plot
- **`explain_instance(X_instance: pd.DataFrame)`**
  - Returns top 5 features affecting a prediction
  - **Parameters:**
    - `X_instance`: Single row DataFrame (without target column)
  - **Returns:** List of (feature, shap_value) tuples, sorted by absolute SHAP value descending
- **`predict(X: pd.DataFrame)`**
  - Makes predictions on new data using optimal threshold
  - **Parameters:**
    - `X`: DataFrame with same features as training data
  - **Returns:** Predictions array
- **`check_drift(X: pd.DataFrame)`**
  - Computes PSI per feature against stored baseline
  - Reports drifted features above configured threshold
  - **Returns:** Tuple of `(drift_report dict, drifted_features dict)`
- **`load(model_path: str, meta_path: str)`**
  - Reloads a previously saved model with full metadata
  - Restores drift baseline, threshold, and SHAP explainer

---

## 🧪 Testing

Run the full test suite:

```bash
cd Self-Trainer
pytest tests/test_stress.py -v
```

The stress test suite covers 122 test cases across 12 test classes:

- NaN / Inf / zero-variance / empty data inputs
- Out-of-range drift data and PSI edge cases
- Degenerate targets (all-same-class, all-zeros, all-ones)
- Constant-probability models in threshold finder
- Rapid-fire experiment logging and registry consistency
- JSON serialization of nan/inf
- Config with `None` / empty-string / numeric modes
- Task detector on empty, all-NaN, bool, float-int series
- Engine pre-fit guard rails (predict/drift/explain before fit all raise)
- Engine fit on duplicate-row, constant-feature, and wide datasets
- Model store save/load with numpy-typed metadata
- Explainer fallback to KernelExplainer for non-tree models

**Expected result:** `122 passed` (warnings about NaN targets and convergence are expected and intentional).

---

## 🖥️ Example Output

Running `main.py` in `full` mode on the Telco Churn dataset produces output like the following:

```
2026-02-19 19:36:19 | INFO | self_trainer.core.config | EngineConfig initialised (mode=full).
2026-02-19 19:36:19 | INFO | self_trainer.engine | Detected task type: classification
2026-02-19 19:36:19 | INFO | self_trainer.engine | Data split — train: 4218, val: 1407, test: 1407 rows.
2026-02-19 19:36:19 | INFO | self_trainer.engine | Cross-validation on TRAIN set (5 folds):
2026-02-19 19:36:21 | INFO | self_trainer.engine |   LogisticRegression   0.846108
2026-02-19 19:36:28 | INFO | self_trainer.engine |   RandomForest         0.827643
2026-02-19 19:36:31 | INFO | self_trainer.engine |   XGBoost              0.831407
2026-02-19 19:36:37 | INFO | self_trainer.engine |   LightGBM             0.826301
2026-02-19 19:37:35 | INFO | self_trainer.engine |   MLP                  0.776735
2026-02-19 19:37:35 | INFO | self_trainer.engine | Tuning LogisticRegression with Optuna...
...
Best params for LogisticRegression: {}
Best CV score after tuning: 0.846108
2026-02-19 19:39:37 | INFO | self_trainer.engine | Tuned LogisticRegression score: 0.846108
2026-02-19 19:39:37 | INFO | self_trainer.engine | Tuning XGBoost with Optuna...
...
Best params for XGBoost: {'n_estimators': 282, 'max_depth': 3, 'learning_rate': 0.017,
                           'subsample': 0.631, 'colsample_bytree': 0.776}
Best CV score after tuning: 0.849974
2026-02-19 19:41:39 | INFO | self_trainer.engine | Tuned XGBoost score: 0.849974
2026-02-19 19:41:39 | INFO | self_trainer.engine | Final best model after tuning: XGBoost
2026-02-19 19:41:39 | INFO | self_trainer.engine | Optimal threshold (validation): 0.3999  |  F1: 0.6724
2026-02-19 19:41:40 | INFO | self_trainer.engine | Test score: 0.828399
2026-02-19 19:41:40 | INFO | self_trainer.engine | Model saved → models/model_20260219_194140.pkl
2026-02-19 19:41:40 | INFO | self_trainer.engine | Metadata saved → models/model_20260219_194140_meta.pkl
2026-02-19 19:41:40 | INFO | self_trainer.engine | Experiment logged → experiments\run_20260219_194140
2026-02-19 19:41:40 | INFO | self_trainer.engine | SHAP explainer fitted on 1000 samples.
2026-02-19 19:41:43 | INFO | self_trainer.engine | Global SHAP plot saved → reports/global_shap.png
```

Key things to note from the log:

- All 5 models are evaluated with 5-fold CV on the train split before any tuning
- Optuna tunes only the **top 2** models (LogisticRegression and XGBoost in this run)
- LogisticRegression had no tunable params defined, so all its Optuna trials return the same score — this is expected behaviour
- XGBoost improved from 0.8314 (baseline CV) to **0.8500** (tuned CV) and became the final best model
- Threshold is optimised on the validation set independently of the test set

---

## 📊 Performance Benchmarks

### Credit Card Fraud Detection (Dataset 1)

| Model              | Cross-Val Score | Performance |
| ------------------ | --------------- | ----------- |
| **XGBoost ⭐**     | **0.990387**    | Outstanding |
| LightGBM           | 0.977915        | Outstanding |
| LogisticRegression | 0.972667        | Excellent   |
| RandomForest       | 0.961487        | Excellent   |
| MLP                | 0.729110        | Poor        |

**Best Model**: XGBoost  
**Final Test Score**: 0.977165  
**Optimal Threshold**: 0.9856  
**Validation F1**: 0.8235

### Telco Customer Churn Detection (Dataset 2)

| Model                     | Cross-Val Score | Performance |
| ------------------------- | --------------- | ----------- |
| **LogisticRegression ⭐** | **0.846108**    | Excellent   |
| XGBoost                   | 0.831407        | Excellent   |
| RandomForest              | 0.827643        | Excellent   |
| LightGBM                  | 0.826301        | Excellent   |
| MLP                       | 0.776735        | Good        |

**Best Model After Tuning**: XGBoost (tuned CV: 0.849974)  
**Final Test Score**: 0.828399  
**Optimal Threshold**: 0.3999  
**Validation F1**: 0.6724

### Top Features (SHAP Analysis — Credit Card Fraud):

1. **V10**: -3.818 (decreases fraud probability)
2. **V14**: -2.056 (decreases fraud probability)
3. **V12**: -1.625 (decreases fraud probability)
4. **V4**: -1.186 (decreases fraud probability)
5. **V16**: -0.730 (decreases fraud probability)

> SHAP global feature importance plot for the Telco Churn run saved to `reports/global_shap.png`.

---

## 🗺️ Roadmap

### v1.0 ✅ (Current)

- [x] Automatic task detection
- [x] Model selection with cross-validation
- [x] Threshold optimization
- [x] SHAP explainability
- [x] Imbalance handling
- [x] Proper train/val/test splitting

### v1.1 ✅

- [x] Structured experiment tracking
- [x] Model versioning & metadata persistence
- [x] Experiment registry system
- [x] Drift detection (PSI-based)
- [x] Reloadable model state
- [x] Validation-based threshold calibration

### v1.2 ✅

- [x] Config-driven engine (dev / full modes)
- [x] Hyperparameter optimization (Optuna)
- [x] Top-2 model tuning (compute-efficient)
- [x] Timeout-based tuning
- [x] Persisted drift baseline storage
- [x] PSI-based feature-level drift reporting
- [x] Expanded metadata persistence (best_params, cv_results, baseline)
- [x] LightGBM stability improvements
- [x] Config-driven tree estimators
- [x] Real-world dataset validation (Credit Card Fraud, Telco Churn)
- [x] Comprehensive stress test suite (122 tests across 12 test classes)
- [x] NaN/Inf/zero-variance robustness for baseline and drift modules
- [x] DriftCalculationError: out-of-range columns excluded from report instead of returning NaN
- [x] Custom exception hierarchy (InvalidConfigError, UnsupportedTaskError, DriftCalculationError, etc.)
- [x] `test_size` and `val_size` parameters exposed on `engine.fit()`
- [x] Configurable bin count for baseline builder

### v2.0 🔮 (Planned)

- [ ] FastAPI deployment wrapper
- [ ] MLflow integration
- [ ] Reinforcement learning model selector
- [ ] Time-series support
- [ ] Web-based monitoring dashboard
- [ ] Docker containerization
- [ ] Kubernetes deployment configs

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### How to Contribute

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/Aryan-20-04/SelfTrainerEngine.git
cd SelfTrainerEngine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install development dependencies
pip install -r requirements.txt

# Run tests
pytest tests/test_stress.py -v        # stress tests (122 cases)
pytest tests/test_self_trainer.py -v  # core tests
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 SelfTrainerEngine Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgments

- Built with [scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/), and [LightGBM](https://lightgbm.readthedocs.io/)
- Explainability powered by [SHAP](https://shap.readthedocs.io/)
- Hyperparameter tuning powered by [Optuna](https://optuna.org/)
- Inspired by AutoML frameworks and production ML best practices

---

## 🐛 Known Issues

- **LogisticRegression Optuna trials are redundant**: When LogisticRegression is selected as a top-2 model, all Optuna trials return the same score because no hyperparameter search space is currently defined for it. The tuner still runs without crashing, and the model is returned unchanged. A parameter grid for `C` and `solver` is planned for v2.0.

If you encounter any other issues, please [open an issue](https://github.com/Aryan-20-04/SelfTrainerEngine/issues) on GitHub.

---

## 📬 Contact

**Project Maintainer**: Aryan Sinha

- GitHub: [@Aryan-20-04](https://github.com/Aryan-20-04)
- Email: sinhaaryan564.@gmail.com
- LinkedIn: [Aryan Sinha](https://www.linkedin.com/in/aryan-sinha-352311328/)

---

## ❓ FAQ

**Q: Can I use custom models?**  
A: Currently, SelfTrainerEngine supports scikit-learn compatible models. Custom model integration is planned for v2.0.

**Q: Does it work with large datasets?**  
A: Yes! The framework is optimized for datasets up to millions of rows. For very large datasets (>10M rows), consider sampling for model selection. Use `mode="dev"` in `EngineConfig` for faster iteration.

**Q: Can I disable certain models?**  
A: Yes, you can modify `core/model_registry.py` to customize which models are evaluated.

**Q: Is GPU support available?**  
A: XGBoost and LightGBM support GPU acceleration. Set `tree_method='gpu_hist'` in model_registry.py.

**Q: How does Optuna tuning work?**  
A: The engine evaluates all candidate models, picks the top 2 by CV score, and tunes only those using Optuna with timeout-based search. Best params are saved in metadata for full reproducibility.

**Q: What is dev vs full mode?**  
A: `mode="dev"` uses fewer CV folds, fewer tree estimators, and a shorter Optuna timeout for rapid iteration. `mode="full"` uses deeper settings for production-quality training.

**Q: Why do all Optuna trials for LogisticRegression return the same value?**  
A: LogisticRegression currently has no hyperparameter search space defined in the tuner, so every trial evaluates the same default config. This is a known limitation — see Known Issues above.

**Q: What do the stress test warnings mean?**  
A: The warnings produced during `pytest tests/test_stress.py` are expected. NaN-target warnings, convergence warnings, and "no positive class" warnings are all intentionally triggered by edge-case tests and are handled gracefully by the framework. All 122 tests pass.

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

Made with ❤️ by the SelfTrainerEngine team

[Report Bug](https://github.com/Aryan-20-04/SelfTrainerEngine/issues) • [Request Feature](https://github.com/Aryan-20-04/SelfTrainerEngine/issues) • [Documentation](https://github.com/Aryan-20-04/SelfTrainerEngine/wiki)

</div>
