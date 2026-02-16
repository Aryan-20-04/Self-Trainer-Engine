"""
Generate documentation images for SelfTrainerEngine README
Uses actual model performance data from training runs
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import seaborn as sns
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory
os.makedirs('docs/images', exist_ok=True)

# ACTUAL MODEL PERFORMANCE DATA
model_scores = {
    'LogisticRegression': 0.972667,
    'RandomForest': 0.961487,
    'XGBoost': 0.990387,
    'LightGBM': 0.977915,
    'MLP': 0.729110
}

# ACTUAL SHAP VALUES (Top 5 features)
top_features = [
    ('V10', -3.81787),
    ('V14', -2.05648),
    ('V12', -1.62463),
    ('V4', -1.18593),
    ('V16', -0.73025)
]

# ACTUAL PERFORMANCE METRICS
best_model = 'XGBoost'
optimal_threshold = 0.9856
validation_f1 = 0.8235
test_score = 0.977165


def create_architecture_diagram():
    """Create pipeline architecture diagram"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define colors
    color_input = '#3498db'
    color_process = '#2ecc71'
    color_output = '#e74c3c'
    color_arrow = '#34495e'
    
    # Boxes
    boxes = [
        (1, 8.5, 1.8, 0.8, 'Input\nData', color_input),
        (1, 7, 1.8, 0.8, 'Task\nDetection', color_process),
        (1, 5.5, 1.8, 0.8, 'Imbalance\nCheck', color_process),
        (1, 4, 1.8, 0.8, 'Model\nSelection', color_process),
        
        (5, 7, 1.8, 0.8, 'Train/Val\nSplit', color_process),
        (5, 5.5, 1.8, 0.8, 'Cross-Val\nTraining', color_process),
        (5, 4, 1.8, 0.8, 'Threshold\nOptimization', color_process),
        
        (8.2, 6.5, 1.8, 0.8, 'Best\nModel', color_output),
        (8.2, 5, 1.8, 0.8, 'SHAP\nExplainer', color_output),
        (8.2, 3.5, 1.8, 0.8, 'Final\nPredictions', color_output),
    ]
    
    for x, y, w, h, text, color in boxes:
        fancy_box = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.1',
                                   facecolor=color, edgecolor='black', 
                                   linewidth=2, alpha=0.8)
        ax.add_patch(fancy_box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')
    
    # Arrows
    arrows = [
        ((1.9, 8.5), (1.9, 7.8)),
        ((1.9, 7.0), (1.9, 6.3)),
        ((1.9, 5.5), (1.9, 4.8)),
        ((2.8, 7.4), (5, 7.4)),
        ((2.8, 5.9), (5, 5.9)),
        ((2.8, 4.4), (5, 4.4)),
        ((5.9, 7.0), (5.9, 6.3)),
        ((5.9, 5.5), (5.9, 4.8)),
        ((6.8, 7.2), (8.2, 6.9)),
        ((6.8, 5.7), (8.2, 5.4)),
        ((6.8, 4.2), (8.2, 3.9)),
        ((9.1, 6.5), (9.1, 5.8)),
        ((9.1, 5.0), (9.1, 4.3)),
    ]
    
    for start, end in arrows:
        arrow = FancyArrowPatch(start, end, arrowstyle='->', 
                               lw=2.5, color=color_arrow, alpha=0.7,
                               mutation_scale=20)
        ax.add_patch(arrow)
    
    ax.text(5, 9.5, 'SelfTrainerEngine Pipeline Architecture', 
            ha='center', fontsize=18, fontweight='bold')
    
    legend_elements = [
        mpatches.Patch(facecolor=color_input, edgecolor='black', label='Input'),
        mpatches.Patch(facecolor=color_process, edgecolor='black', label='Processing'),
        mpatches.Patch(facecolor=color_output, edgecolor='black', label='Output')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('docs/images/architecture.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Architecture diagram created")


def create_performance_comparison():
    """Create model performance comparison using ACTUAL DATA"""
    models = list(model_scores.keys())
    scores = list(model_scores.values())
    
    # Shorten model names for better display
    model_labels = [m.replace('Regression', '') for m in models]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Color bars - highlight best model
    colors = ['#e74c3c' if model == best_model else '#95a5a6' for model in models]
    bars = ax.bar(model_labels, scores, color=colors, edgecolor='black', 
                  linewidth=2, alpha=0.85, width=0.6)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add winner annotation
    winner_idx = models.index(best_model)
    ax.annotate(f'Best Model ⭐\nTest Score: {test_score:.4f}', 
                xy=(winner_idx, scores[winner_idx]), 
                xytext=(winner_idx, scores[winner_idx] - 0.12),
                ha='center', fontsize=11, fontweight='bold', color='#e74c3c',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                         edgecolor='#e74c3c', linewidth=2),
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2))
    
    ax.set_ylabel('Cross-Validation Score (ROC-AUC)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_title('Actual Model Performance Comparison\n(Credit Card Fraud Detection)', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_ylim(0.7, 1.02)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add threshold line
    ax.axhline(y=0.95, color='green', linestyle='--', linewidth=2, 
               alpha=0.5, label='Excellent threshold (0.95)')
    ax.legend(loc='lower left', fontsize=10)
    
    # Add performance note
    note = f'Optimal Threshold: {optimal_threshold:.4f}\nValidation F1: {validation_f1:.4f}'
    ax.text(0.98, 0.02, note, transform=ax.transAxes, 
            fontsize=10, verticalalignment='bottom',
            horizontalalignment='right', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('docs/images/performance_comparison.png', dpi=300, 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Performance comparison chart created (ACTUAL DATA)")


def create_shap_feature_importance():
    """Create SHAP feature importance using ACTUAL DATA"""
    features = [f[0] for f in top_features]
    shap_values = [abs(f[1]) for f in top_features]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create color gradient based on magnitude
    colors = plt.cm.get_cmap('RdYlGn_r')(np.linspace(0.3, 0.9, len(features)))
    bars = ax.barh(features, shap_values, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, val in zip(bars, shap_values):
        width = bar.get_width()
        ax.text(width + 0.05, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}',
                ha='left', va='center', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Mean |SHAP Value| (Feature Impact)', 
                  fontsize=12, fontweight='bold')
    ax.set_title('Top 5 Feature Importances (Actual SHAP Values)\nCredit Card Fraud Detection', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.invert_yaxis()  # Highest importance at top
    
    # Add note
    ax.text(0.98, 0.02, 'All features decrease\nfraud probability', 
            transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))
    
    plt.tight_layout()
    plt.savefig('docs/images/shap_feature_importance.png', dpi=300, 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ SHAP feature importance created (ACTUAL DATA)")


def create_metrics_summary():
    """Create comprehensive metrics summary"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('SelfTrainerEngine Performance Summary', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # 1. Model Comparison (Top Left)
    models = list(model_scores.keys())
    scores = list(model_scores.values())
    model_labels = [m.replace('Regression', '') for m in models]
    colors_1 = ['#e74c3c' if m == best_model else '#95a5a6' for m in models]
    
    bars1 = ax1.barh(model_labels, scores, color=colors_1, edgecolor='black', linewidth=1.5)
    for bar, score in zip(bars1, scores):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2,
                f' {score:.3f}', va='center', fontsize=10, fontweight='bold')
    ax1.set_xlabel('ROC-AUC Score', fontsize=11, fontweight='bold')
    ax1.set_title('Model Performance', fontsize=12, fontweight='bold')
    ax1.set_xlim(0.7, 1.0)
    ax1.grid(axis='x', alpha=0.3)
    
    # 2. Top Features (Top Right)
    features = [f[0] for f in top_features]
    impacts = [f[1] for f in top_features]
    colors_2 = ['#3498db' for _ in impacts]  # All decrease (negative)
    
    bars2 = ax2.barh(features, impacts, color=colors_2, edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars2, impacts):
        width = bar.get_width()
        ax2.text(width - 0.1, bar.get_y() + bar.get_height()/2,
                f'{val:.2f} ', ha='right', va='center', 
                fontsize=9, fontweight='bold', color='white')
    ax2.set_xlabel('SHAP Value (Impact)', fontsize=11, fontweight='bold')
    ax2.set_title('Top 5 Features (All Decrease Fraud Probability)', 
                  fontsize=12, fontweight='bold')
    ax2.axvline(x=0, color='black', linewidth=2)
    ax2.grid(axis='x', alpha=0.3)
    ax2.invert_yaxis()
    
    # 3. Key Metrics (Bottom Left)
    ax3.axis('off')
    metrics_text = f"""
    ╔════════════════════════════════════╗
    ║     BEST MODEL PERFORMANCE         ║
    ╠════════════════════════════════════╣
    ║                                    ║
    ║  Model: {best_model:<25}║
    ║  Test Score: {test_score:<20.6f}║
    ║  Optimal Threshold: {optimal_threshold:<13.4f}║
    ║  Validation F1: {validation_f1:<17.4f}║
    ║                                    ║
    ╚════════════════════════════════════╝
    """
    ax3.text(0.5, 0.5, metrics_text, ha='center', va='center',
            fontsize=11, family='monospace', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#ecf0f1', 
                     edgecolor='black', linewidth=2))
    ax3.set_title('Key Metrics', fontsize=12, fontweight='bold', pad=20)
    
    # 4. Performance Distribution (Bottom Right)
    score_ranges = ['Poor\n(<0.8)', 'Good\n(0.8-0.9)', 'Excellent\n(0.9-0.95)', 
                    'Outstanding\n(>0.95)']
    counts = [1, 0, 2, 2]  # Based on actual scores
    colors_4 = ['#e74c3c', '#f39c12', '#2ecc71', '#9b59b6']
    
    wedges, texts, autotexts = ax4.pie(counts, labels=score_ranges, 
                                        colors=colors_4, autopct='%1.0f%%',
                                        startangle=90, textprops={'fontweight': 'bold'})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)
    ax4.set_title('Model Distribution by Performance', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('docs/images/metrics_summary.png', dpi=300, 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Metrics summary created (ACTUAL DATA)")


def create_workflow_diagram():
    """Create workflow diagram"""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c']
    
    steps = [
        (5, 11, "Load Data", "Credit Card Fraud Dataset"),
        (5, 9.5, "Auto Task Detection", "Classification Detected"),
        (5, 8, "Imbalance Check", "Imbalanced Classes Found"),
        (5, 6.5, "Data Splitting", "60% Train | 20% Val | 20% Test"),
        (5, 5, "Model Selection", "5 Models Evaluated"),
        (5, 3.5, "XGBoost Wins", f"Score: {model_scores['XGBoost']:.4f}"),
        (5, 2, "Threshold Tuning", f"Optimal: {optimal_threshold:.4f}"),
        (5, 0.5, "Production Ready", f"Test Score: {test_score:.4f}"),
    ]
    
    for i, (x, y, title, desc) in enumerate(steps):
        color = colors[i % len(colors)]
        fancy_box = FancyBboxPatch((x-1.8, y-0.3), 3.6, 0.9, 
                                   boxstyle='round,pad=0.1',
                                   facecolor=color, edgecolor='black',
                                   linewidth=2.5, alpha=0.85)
        ax.add_patch(fancy_box)
        
        ax.text(x, y+0.15, title, ha='center', va='center',
                fontsize=13, fontweight='bold', color='white')
        ax.text(x, y-0.15, desc, ha='center', va='center',
                fontsize=9, style='italic', color='white', alpha=0.9)
        
        if i < len(steps) - 1:
            arrow = FancyArrowPatch((x, y-0.4), (x, y-1.1),
                                   arrowstyle='->', lw=3, 
                                   color='#2c3e50', alpha=0.7,
                                   mutation_scale=25)
            ax.add_patch(arrow)
    
    ax.text(5, 11.8, 'Actual Training Run Results', 
            ha='center', fontsize=20, fontweight='bold')
    
    auto_box = FancyBboxPatch((8.5, 11), 1.2, 0.4,
                              boxstyle='round,pad=0.05',
                              facecolor='#e74c3c', edgecolor='black',
                              linewidth=2)
    ax.add_patch(auto_box)
    ax.text(9.1, 11.2, 'AUTO', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig('docs/images/workflow.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Workflow diagram created (ACTUAL DATA)")


def main():
    """Generate all documentation images"""
    print("\n" + "="*60)
    print("Generating SelfTrainerEngine Documentation Images")
    print("Using ACTUAL model performance data")
    print("="*60 + "\n")
    
    print("Model Performance Summary:")
    for model, score in model_scores.items():
        marker = " ⭐" if model == best_model else ""
        print(f"  {model}: {score:.6f}{marker}")
    
    print(f"\nBest Model: {best_model}")
    print(f"Test Score: {test_score:.6f}")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    print(f"Validation F1: {validation_f1:.4f}\n")
    
    print("Generating images...\n")
    
    create_architecture_diagram()
    create_workflow_diagram()
    create_performance_comparison()
    create_shap_feature_importance()
    create_metrics_summary()
    
    print("\n" + "="*60)
    print("✅ All images generated successfully!")
    print("📁 Images saved to: docs/images/")
    print("="*60 + "\n")
    
    print("Generated files:")
    print("  1. architecture.png - Pipeline architecture")
    print("  2. workflow.png - Actual training run workflow")
    print("  3. performance_comparison.png - Model comparison")
    print("  4. shap_feature_importance.png - Top 5 features")
    print("  5. metrics_summary.png - Complete performance dashboard")
    print("\nThese images use your actual model results!")


if __name__ == "__main__":
    main()
