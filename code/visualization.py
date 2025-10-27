import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def plot_pareto_front(pareto_results, filename='pareto_front.png', dataset_name=''):
    """Plot Pareto front"""
    acc = [res['accuracy'] for res in pareto_results]
    num_feats = [res['num_features'] for res in pareto_results]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(num_feats, acc, c='blue', s=100, alpha=0.6, edgecolors='black')
    plt.xlabel('Number of Features', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Pareto Front - {dataset_name}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"  Pareto front plot saved: {filename}")

def plot_confusion_matrix(cm, filename='confusion_matrix.png', title=''):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'Confusion Matrix - {title}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"  Confusion matrix saved: {filename}")

def plot_comparison_bar(metrics_dict, filename='comparison.png', dataset_name=''):
    """Bar chart comparing methods"""
    methods = list(metrics_dict.keys())
    accuracies = [metrics_dict[m]['accuracy'] for m in methods]
    f1_scores = [metrics_dict[m]['f1_score'] for m in methods]
    num_features = [metrics_dict[m]['num_features'] for m in methods]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Accuracy
    axes[0].bar(methods, accuracies, color=['gray', 'orange', 'green'], alpha=0.7)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
    axes[0].set_ylim([min(accuracies) - 0.1, 1.0])
    
    # F1-Score
    axes[1].bar(methods, f1_scores, color=['gray', 'orange', 'green'], alpha=0.7)
    axes[1].set_ylabel('F1-Score', fontsize=12)
    axes[1].set_title('F1-Score Comparison', fontsize=12, fontweight='bold')
    axes[1].set_ylim([min(f1_scores) - 0.1, 1.0])
    
    # Number of Features
    axes[2].bar(methods, num_features, color=['gray', 'orange', 'green'], alpha=0.7)
    axes[2].set_ylabel('Number of Features', fontsize=12)
    axes[2].set_title('Feature Count', fontsize=12, fontweight='bold')
    
    plt.suptitle(f'Comparison - {dataset_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"  Comparison chart saved: {filename}")

def export_results_to_csv(all_results, filename='results_summary.csv'):
    """Export all results to CSV"""
    df = pd.DataFrame(all_results)
    df.to_csv(filename, index=False)
    print(f"  Results exported to: {filename}")
