"""
ML Prediction Plotting Utilities
Generates plots for email reports showing model accuracy
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import base64
from io import BytesIO


def create_prediction_plot(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           y_pred_proba: Optional[np.ndarray] = None,
                           title: str = "Predictions vs Actual") -> str:
    """
    Create prediction vs actual plot and return as base64 string for email embedding.

    Args:
        y_true: Actual labels (binary: 0/1)
        y_pred: Predicted labels (binary: 0/1)
        y_pred_proba: Predicted probabilities (0-1), optional
        title: Plot title

    Returns:
        Base64 encoded PNG image string
    """
    fig, axes = plt.subplots(1, 2 if y_pred_proba is not None else 1, figsize=(14, 5))

    if y_pred_proba is None:
        axes = [axes]

    # Plot 1: Confusion Matrix
    from sklearn.metrics import confusion_matrix, accuracy_score
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title(f'{title}\nAccuracy: {accuracy:.1%}')
    axes[0].set_ylabel('Actual')
    axes[0].set_xlabel('Predicted')
    axes[0].set_xticklabels(['Loss', 'Win'])
    axes[0].set_yticklabels(['Loss', 'Win'])

    # Plot 2: Probability Distribution (if available)
    if y_pred_proba is not None:
        axes[1].hist(y_pred_proba[y_true == 0], bins=20, alpha=0.5, label='Actual Loss', color='red')
        axes[1].hist(y_pred_proba[y_true == 1], bins=20, alpha=0.5, label='Actual Win', color='green')
        axes[1].axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold')
        axes[1].set_xlabel('Predicted Probability')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Prediction Probability Distribution')
        axes[1].legend()

    plt.tight_layout()

    # Convert to base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)

    return image_base64


def create_roc_curve_plot(y_true: np.ndarray,
                          y_pred_proba: np.ndarray,
                          auc_score: float) -> str:
    """
    Create ROC curve plot.

    Args:
        y_true: Actual labels
        y_pred_proba: Predicted probabilities
        auc_score: AUC score

    Returns:
        Base64 encoded PNG image string
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve - Model Performance')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)

    return image_base64


def create_calibration_plot(y_true: np.ndarray,
                            y_pred_proba: np.ndarray,
                            n_bins: int = 10) -> str:
    """
    Create calibration plot showing if predicted probabilities match actual outcomes.

    Args:
        y_true: Actual labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins for calibration

    Returns:
        Base64 encoded PNG image string
    """
    from sklearn.calibration import calibration_curve

    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model Calibration')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')

    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Actual Win Rate')
    ax.set_title('Calibration Curve - Are Probabilities Accurate?')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)

    return image_base64


def create_feature_importance_plot(feature_names: List[str],
                                   importance_values: np.ndarray,
                                   top_n: int = 15) -> str:
    """
    Create feature importance bar plot.

    Args:
        feature_names: List of feature names
        importance_values: Importance scores
        top_n: Show top N features

    Returns:
        Base64 encoded PNG image string
    """
    # Sort by importance
    indices = np.argsort(importance_values)[-top_n:][::-1]
    top_features = [feature_names[i] for i in indices]
    top_importance = importance_values[indices]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    ax.barh(range(len(top_features)), top_importance, color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Top {top_n} Most Important Features')
    ax.invert_yaxis()

    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)

    return image_base64


def create_performance_timeline(dates: List[str],
                               actual: List[int],
                               predicted: List[int]) -> str:
    """
    Create timeline showing predictions vs actual over time.

    Args:
        dates: List of dates (strings)
        actual: Actual outcomes
        predicted: Predicted outcomes

    Returns:
        Base64 encoded PNG image string
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    x = range(len(dates))

    # Plot actual and predicted
    ax.scatter(x, actual, label='Actual', alpha=0.6, s=50, c='blue')
    ax.scatter(x, predicted, label='Predicted', alpha=0.6, s=50, c='red', marker='x')

    # Show correct/incorrect predictions
    correct = [actual[i] == predicted[i] for i in range(len(actual))]
    accuracy = sum(correct) / len(correct) * 100

    ax.set_xlabel('Trade Index')
    ax.set_ylabel('Outcome (0=Loss, 1=Win)')
    ax.set_title(f'Predictions vs Actual Over Time (Accuracy: {accuracy:.1f}%)')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Loss', 'Win'])
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)

    return image_base64


def create_experiment_comparison_plot(experiment_results: dict) -> str:
    """
    Create comparison plot for multiple experiments.

    Args:
        experiment_results: Dict with {experiment_name: {'auc': float, 'accuracy': float}}

    Returns:
        Base64 encoded PNG image string
    """
    experiments = list(experiment_results.keys())
    aucs = [experiment_results[exp]['auc'] for exp in experiments]
    accuracies = [experiment_results[exp].get('accuracy', 0) for exp in experiments]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(experiments))
    width = 0.35

    ax.bar(x - width/2, aucs, width, label='AUC', color='skyblue')
    ax.bar(x + width/2, accuracies, width, label='Accuracy', color='lightcoral')

    ax.set_xlabel('Experiment')
    ax.set_ylabel('Score')
    ax.set_title('Experiment Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)

    return image_base64
