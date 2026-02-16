import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Optional, Union


class ModelExplainer:

    def __init__(self, model):
        self.model = model
        self.explainer: Optional[Union[shap.TreeExplainer, shap.KernelExplainer]] = None

    def fit(self, X_sample):
        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception:
            self.explainer = shap.KernelExplainer(
                self.model.predict, X_sample
            )

    def global_explanation(self, X_sample, save_path=None):
        if self.explainer is None:
            raise RuntimeError("Explainer not initialized. Call fit() first.")

        shap_values = self.explainer.shap_values(X_sample)
        shap_values = np.array(shap_values)

        # Handle multi-class case (keep class 1 for binary classification)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        plt.figure()
        shap.summary_plot(shap_values, X_sample, show=False)

        if save_path:
            os.makedirs("reports", exist_ok=True)
            filepath = f"reports/{save_path}"
            plt.savefig(filepath, bbox_inches="tight")
            plt.close()
            return filepath

        plt.show()

    def explain_instance(self, X_instance):
        if self.explainer is None:
            raise RuntimeError("Explainer not initialized. Call fit() first.")

        shap_values = self.explainer.shap_values(X_instance)
        
        # Handle different return types
        if isinstance(shap_values, list):
            # Multi-class: take positive class for binary, or first class
            shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
        
        shap_values = np.array(shap_values)
        
        # Get values for single instance
        if shap_values.ndim == 2:
            shap_values_instance = shap_values[0]
        elif shap_values.ndim == 1:
            shap_values_instance = shap_values
        else:
            raise ValueError(f"Unexpected shap_values shape: {shap_values.shape}")

        contributions = list(zip(X_instance.columns, shap_values_instance))
        contributions = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)[:5]
        
        return contributions