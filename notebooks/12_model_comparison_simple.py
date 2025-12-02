#!/usr/bin/env python3
"""
Simple Model Comparison using Existing Results
Compares baseline evaluation metrics with GNN training metrics.
Note: These are different evaluation tasks (recommendation vs link prediction).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_RESULTS_DIR = PROJECT_ROOT / 'results' / 'evaluation'
GNN_RESULTS_DIR = PROJECT_ROOT / 'results' / 'GNN'
COMPARISON_RESULTS_DIR = PROJECT_ROOT / 'results' / 'model_comparison'
COMPARISON_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_existing_results():
    """Load existing evaluation and training results."""
    print("Loading existing results...")
    
    # Load baseline evaluation metrics
    baseline_metrics = {}
    try:
        eval_df = pd.read_csv(EVAL_RESULTS_DIR / 'evaluation_metrics.csv')
        print("  Loaded baseline evaluation metrics")
        
        # Get Study Buddy metrics
        sb_row = eval_df[eval_df['Unnamed: 0'] == 'Study_Buddy'].iloc[0]
        baseline_metrics = {
            'Precision@5': sb_row['Precision@5'],
            'Precision@10': sb_row['Precision@10'],
            'Recall@5': sb_row['Recall@5'],
            'Recall@10': sb_row['Recall@10'],
            'F1@5': sb_row['F1@5'],
            'F1@10': sb_row['F1@10'],
            'MRR': sb_row['MRR'],
            'Coverage': sb_row['Coverage'],
            'Diversity': sb_row['Average_Intra_List_Diversity']
        }
    except Exception as e:
        print(f"  Warning: Could not load baseline metrics: {e}")
    
    # GNN training metrics (from training visualizations/plots)
    # These are link prediction metrics from the actual training
    gnn_metrics = {
        'ROC_AUC': 0.986,  # From ROC.png - Final test AUC
        'Validation_AUC': 0.745,  # From Validation_AUC_Hetero_GraphSage.png - Peak validation AUC
        'Average_Precision': 0.973,  # From AUC_AP.png - Validation AP at epoch 50
        'Train_AUC': 0.979,  # From AUC_AP.png - Train AUC at epoch 50
        'Train_AP': 0.981,  # From AUC_AP.png - Train AP at epoch 50
        'Final_Loss': 0.438,  # From BPR_Loss_Hetero_GraphSage.png - Final BPR loss
        'BCE_Loss_Train': 0.515,  # From BCE_Loss.png - Final train BCE loss
        'BCE_Loss_Val': 0.52,  # From BCE_Loss.png - Final validation BCE loss
        'Note': 'GNN was trained for link prediction (predicting edges in graph)'
    }
    
    return baseline_metrics, gnn_metrics


def create_comparison_report(baseline_metrics, gnn_metrics):
    """Create a comparison report explaining the differences."""
    print("\n" + "="*60)
    print("MODEL COMPARISON REPORT")
    print("="*60)
    
    print("\nBASELINE MODEL (Embedding-based):")
    print("  Task: Recommendation Quality")
    print("  Metrics:")
    for metric, value in baseline_metrics.items():
        if metric != 'Note':
            print(f"    {metric:25s}: {value:.4f}")
    
    print("\nGNN MODEL (GraphSAGE):")
    print("  Task: Link Prediction (Graph Edge Prediction)")
    print("  Metrics:")
    print(f"    ROC AUC (Test): {gnn_metrics['ROC_AUC']:.4f}")
    print(f"    Validation AUC: {gnn_metrics['Validation_AUC']:.4f}")
    print(f"    Average Precision: {gnn_metrics['Average_Precision']:.4f}")
    print(f"    Train AUC: {gnn_metrics['Train_AUC']:.4f}")
    print(f"    Train AP: {gnn_metrics['Train_AP']:.4f}")
    print(f"    Final BPR Loss: {gnn_metrics['Final_Loss']:.4f}")
    print(f"    Final BCE Loss (Train/Val): {gnn_metrics['BCE_Loss_Train']:.4f} / {gnn_metrics['BCE_Loss_Val']:.4f}")
    print(f"    Note: {gnn_metrics['Note']}")
    
    print("\n" + "="*60)
    print("IMPORTANT NOTE:")
    print("="*60)
    print("These models are evaluated on DIFFERENT tasks:")
    print("  • Baseline: Recommendation quality (Precision@k, Recall@k)")
    print("  • GNN: Link prediction (AUC - predicting if two users connect)")
    print("\nThey cannot be directly compared because:")
    print("  • GNN optimizes for predicting edges in the graph")
    print("  • Baseline optimizes for recommendation relevance")
    print("  • Different objectives = different metrics")
    
    return {
        'baseline': baseline_metrics,
        'gnn': gnn_metrics,
        'comparison_note': 'Different evaluation tasks - not directly comparable'
    }


def create_visualization(baseline_metrics, gnn_metrics, save_path):
    """Create visualization showing both models' strengths."""
    print("\nCreating comparison visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Comparison: Baseline vs GNN', fontsize=16, fontweight='bold')
    
    # 1. Baseline Recommendation Metrics
    metrics_to_plot = ['Precision@5', 'Recall@5', 'F1@5', 'MRR']
    values = [baseline_metrics.get(m, 0) for m in metrics_to_plot]
    
    axes[0, 0].bar(metrics_to_plot, values, alpha=0.8, color='green', edgecolor='black')
    axes[0, 0].set_title('Baseline: Recommendation Quality Metrics', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Score', fontsize=10)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    for i, (metric, val) in enumerate(zip(metrics_to_plot, values)):
        axes[0, 0].text(i, val + 0.02, f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. GNN Link Prediction Metrics
    gnn_metrics_list = ['ROC AUC', 'Val AUC', 'Avg Precision']
    gnn_values = [gnn_metrics['ROC_AUC'], gnn_metrics['Validation_AUC'], gnn_metrics['Average_Precision']]
    
    axes[0, 1].bar(gnn_metrics_list, gnn_values, alpha=0.8, color='purple', edgecolor='black')
    axes[0, 1].set_title('GNN: Link Prediction Metrics', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Score', fontsize=10)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    for i, (metric, val) in enumerate(zip(gnn_metrics_list, gnn_values)):
        axes[0, 1].text(i, val + 0.01, f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Coverage and Diversity
    coverage_diversity = {
        'Coverage': baseline_metrics.get('Coverage', 0),
        'Diversity': baseline_metrics.get('Diversity', 0)
    }
    axes[1, 0].bar(coverage_diversity.keys(), coverage_diversity.values(), 
                   alpha=0.8, color='orange', edgecolor='black')
    axes[1, 0].set_title('Baseline: Coverage & Diversity', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Score', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    for metric, val in coverage_diversity.items():
        axes[1, 0].text(metric, val + 0.01, f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 4. Summary text
    axes[1, 1].axis('off')
    summary_text = """
    COMPARISON SUMMARY
    
    Baseline Model:
    • Task: Recommendation Quality
    • Optimizes: Precision, Recall, MRR
    • Best for: Ranking relevant recommendations
    
    GNN Model:
    • Task: Link Prediction  
    • Optimizes: AUC (edge prediction)
    • Best for: Predicting connections
    
    Key Insight:
    These models serve different purposes:
    • Baseline: "What should I recommend?"
    • GNN: "Will these users connect?"
    
    Both are valuable but measure different
    aspects of the recommendation system.
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, 
                    verticalalignment='center', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(save_path / 'model_comparison_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Visualization saved to: {save_path / 'model_comparison_summary.png'}")


def main():
    """Main function."""
    print("="*60)
    print("SIMPLE MODEL COMPARISON (Using Existing Results)")
    print("="*60)
    
    # Load existing results
    baseline_metrics, gnn_metrics = load_existing_results()
    
    # Create comparison report
    comparison = create_comparison_report(baseline_metrics, gnn_metrics)
    
    # Save results
    with open(COMPARISON_RESULTS_DIR / 'simple_comparison_results.pkl', 'wb') as f:
        pickle.dump(comparison, f)
    
    # Create comparison DataFrame
    comparison_data = []
    for metric in ['Precision@5', 'Precision@10', 'Recall@5', 'Recall@10', 'F1@5', 'F1@10', 'MRR', 'Coverage', 'Diversity']:
        comparison_data.append({
            'Metric': metric,
            'Baseline': baseline_metrics.get(metric, 'N/A'),
            'GNN': 'N/A (Different Task)',
            'Note': 'Baseline: Recommendation | GNN: Link Prediction'
        })
    
    # Add GNN metrics
    for metric in ['ROC_AUC', 'Validation_AUC', 'Average_Precision', 'Train_AUC', 'Train_AP']:
        comparison_data.append({
            'Metric': metric,
            'Baseline': 'N/A (Different Task)',
            'GNN': gnn_metrics.get(metric, 'N/A'),
            'Note': 'GNN: Link Prediction | Baseline: Recommendation'
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(COMPARISON_RESULTS_DIR / 'simple_comparison_results.csv', index=False)
    
    # Create visualization
    create_visualization(baseline_metrics, gnn_metrics, COMPARISON_RESULTS_DIR)
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETED!")
    print("="*60)
    print(f"Results saved to: {COMPARISON_RESULTS_DIR}")
    print("\nNote: This comparison shows that baseline and GNN")
    print("      are optimized for different tasks and cannot")
    print("      be directly compared using the same metrics.")


if __name__ == "__main__":
    main()

