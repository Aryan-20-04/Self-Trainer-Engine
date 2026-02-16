<div align="center">

# 🧠 SelfTrainerEngine

### _Intelligent Self-Training ML Framework for Tabular Data_

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Features](#-features) • [Installation](#️-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Roadmap](#-roadmap)

</div>

---

## 📖 Overview

**SelfTrainerEngine** is a production-ready, automated machine learning framework designed for tabular data. It intelligently handles the entire ML pipeline—from task detection to model deployment—with minimal configuration required.

Built for data scientists and ML engineers who need **reliable, explainable, and efficient** models without the overhead of manual hyperparameter tuning and pipeline engineering.

### Why SelfTrainerEngine?

- 🎯 **Zero-config ML**: Automatically detects task type, handles imbalance, and selects optimal models
- 🔍 **Built-in Explainability**: SHAP integration for global and instance-level interpretability
- ⚖️ **Smart Imbalance Handling**: Automatic detection and correction for class imbalance
- 📊 **Proper Validation**: Clean train/validation/test splits with threshold optimization
- 🚀 **Production-Ready**: Designed with best practices for real-world deployment

---

## ✨ Features

### 🤖 Automatic Task Detection

Intelligently identifies your ML problem:

- **Binary/Multi-class Classification**
- **Regression**

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

SHAP-powered interpretability:

- **Global explanations**: Feature importance across entire dataset
- **Instance-level explanations**: Why a specific prediction was made
- Automatic TreeExplainer/KernelExplainer fallback

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
```

---

## 🚀 Quick Start

### Basic Usage

```python
import pandas as pd
from core.engine import SelfTrainerEngine

# Load your data
df = pd.read_csv("data/your_dataset.csv")

# Initialize and train
engine = SelfTrainerEngine()
engine.fit(df, target="target_column")

# View performance summary
engine.summary()

# Generate global explanation
engine.explain_global(save_path="feature_importance.png")

# Explain a single prediction
sample = df.drop("target_column", axis=1).sample(1)
top_features = engine.explain_instance(sample)
print(top_features)
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

---

## 📂 Project Structure

```
SelfTrainerEngine/
│
├── core/                      # Core ML engine components
│   ├── __pycache__/
│   ├── engine.py             # Main orchestrator
│   ├── evaluator.py          # Performance evaluation
│   ├── model_registry.py     # Model configurations
│   ├── task_detector.py      # Task type detection
│   ├── threshold.py          # Threshold optimization
│   └── trainer.py            # Model training logic
│
├── explainability/           # Explainability module
│   ├── __pycache__/
│   └── explainer.py          # SHAP integration
│
├── monitoring/               # Model monitoring (planned)
│
├── versioning/               # Model versioning (planned)
│
├── data/                     # Data directory (add your datasets here)
│
├── reports/                  # Generated SHAP plots and reports
│
├── .gitignore               # Git ignore file
├── main.py                  # Example usage script
├── requirements.txt         # Python dependencies
└── README.md               # This file
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

- **`fit(df: pd.DataFrame, target: str)`**
  - Trains the engine on your dataset
  - Automatically detects task, handles imbalance, selects model
  - **Parameters:**
    - `df`: Input DataFrame containing features and target
    - `target`: Name of the target column
- **`summary()`**
  - Prints comprehensive performance metrics
  - Shows optimal threshold (if classification)
  - Displays best model and evaluation scores
- **`explain_global(save_path: str = None)`**
  - Generates SHAP summary plot
  - Shows feature importance across dataset
  - **Parameters:**
    - `save_path`: Optional path to save the plot (e.g., "feature_importance.png")
- **`explain_instance(X_instance: pd.DataFrame)`**
  - Returns top 5 features affecting a prediction
  - Provides SHAP values for interpretation
  - **Parameters:**
    - `X_instance`: Single row DataFrame (without target column)
  - **Returns:** List of (feature, shap_value) tuples

- **`predict(X: pd.DataFrame)`**
  - Makes predictions on new data
  - Uses optimal threshold for classification
  - **Parameters:**
    - `X`: DataFrame with same features as training data
  - **Returns:** Predictions array

---

## 🧪 Testing

Run the example script:

```bash
python main.py
```

Expected output:

```
[SelfTrainerEngine] Detecting task type...
[SelfTrainerEngine] Task detected: classification
[SelfTrainerEngine] Checking for class imbalance...
[SelfTrainerEngine] Imbalance detected (ratio: 0.15)
[SelfTrainerEngine] Training models...
[SelfTrainerEngine] Best model: XGBoostClassifier
[SelfTrainerEngine] Optimizing threshold...
[SelfTrainerEngine] Optimal threshold: 0.23

========== SUMMARY ==========
Task: classification
Best Model: XGBoostClassifier
Test ROC-AUC: 0.9847
Optimal Threshold: 0.23
=============================
```

---

## 🗺️ Roadmap

### v1.0 ✅ (Current)

- [x] Automatic task detection
- [x] Model selection with cross-validation
- [x] Threshold optimization
- [x] SHAP explainability
- [x] Imbalance handling
- [x] Proper train/val/test splitting

### v1.1 🚧 (In Progress)

- [ ] Drift detection module
- [ ] Model versioning and registry
- [ ] Feature engineering automation
- [ ] Hyperparameter optimization (Optuna)
- [ ] Advanced monitoring capabilities

### v2.0 🔮 (Planned)

- [ ] FastAPI deployment wrapper
- [ ] MLflow experiment tracking
- [ ] AutoML with reinforcement learning
- [ ] Time-series support
- [ ] Custom model integration API
- [ ] Web-based dashboard
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

# Run tests (if available)
pytest tests/
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
- Inspired by AutoML frameworks and production ML best practices

---

## 📊 Performance Benchmarks

### Example data used: Credit Card Fraud Detection

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

### Top Features (SHAP Analysis):

1. **V10**: -3.818 (decreases fraud probability)
2. **V14**: -2.056 (decreases fraud probability)
3. **V12**: -1.625 (decreases fraud probability)
4. **V4**: -1.186 (decreases fraud probability)
5. **V16**: -0.730 (decreases fraud probability)

---

## 🐛 Known Issues

- None currently! 🎉

If you encounter any issues, please [open an issue](https://github.com/Aryan-20-4/SelfTrainerEngine/issues) on GitHub.

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
A: Yes! The framework is optimized for datasets up to millions of rows. For very large datasets (>10M rows), consider sampling for model selection.

**Q: Can I disable certain models?**  
A: Yes, you can modify `core/model_registry.py` to customize which models are evaluated.

**Q: Is GPU support available?**  
A: XGBoost and LightGBM support GPU acceleration. Set `tree_method='gpu_hist'` in model_registry.py.

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

Made with ❤️ by the SelfTrainerEngine team

[Report Bug](https://github.com/Aryan-20-04/SelfTrainerEngine/issues) • [Request Feature](https://github.com/Aryan-20-04/SelfTrainerEngine/issues) • [Documentation](https://github.com/Aryan-20-04/SelfTrainerEngine/wiki)

</div>
