"""
Create beautiful training visualization plots
Run after training: python plot_training.py
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

def load_history(model_name):
    """Load training history from JSON"""
    history_path = f'results/{model_name}_history.json'
    with open(history_path, 'r') as f:
        return json.load(f)

def plot_all_metrics(models=['resnet50', 'efficientnet']):
    """Create comprehensive training plots"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Dynamics Comparison', fontsize=16, fontweight='bold')
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    for idx, model_name in enumerate(models):
        try:
            history = load_history(model_name)
            color = colors[idx]
            label = model_name.upper().replace('_', ' ')
            
            epochs = history['epoch']
            
            # Plot 1: Training & Validation Loss
            ax = axes[0, 0]
            ax.plot(epochs, history['train_loss'], 
                   label=f'{label} - Train', 
                   color=color, linewidth=2, alpha=0.8)
            ax.plot(epochs, history['val_loss'], 
                   label=f'{label} - Val', 
                   color=color, linewidth=2, linestyle='--', alpha=0.8)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training & Validation Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 2: Learning Rate Schedule
            ax = axes[0, 1]
            ax.plot(epochs, history['learning_rate'], 
                   label=label, color=color, linewidth=2, marker='o', markersize=4)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 3: Gradient Norm
            ax = axes[0, 2]
            ax.plot(epochs, history['grad_norm'], 
                   label=label, color=color, linewidth=2, alpha=0.8)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Gradient Norm')
            ax.set_title('Gradient Norm (Training Stability)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except FileNotFoundError:
            print(f"Warning: History file not found for {model_name}")
            continue
    
    # Plot 4: Loss Improvement Over Time
    ax = axes[1, 0]
    for idx, model_name in enumerate(models):
        try:
            history = load_history(model_name)
            color = colors[idx]
            label = model_name.upper().replace('_', ' ')
            
            val_losses = history['val_loss']
            improvement = [(val_losses[0] - loss) / val_losses[0] * 100 
                          for loss in val_losses]
            
            ax.plot(history['epoch'], improvement, 
                   label=label, color=color, linewidth=2, marker='o', markersize=4)
        except:
            continue
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Improvement (%)')
    ax.set_title('Validation Loss Improvement from Start')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # Plot 5: Generalization Gap (Train - Val Loss)
    ax = axes[1, 1]
    for idx, model_name in enumerate(models):
        try:
            history = load_history(model_name)
            color = colors[idx]
            label = model_name.upper().replace('_', ' ')
            
            gap = [t - v for t, v in zip(history['train_loss'], history['val_loss'])]
            ax.plot(history['epoch'], gap, 
                   label=label, color=color, linewidth=2, alpha=0.8)
        except:
            continue
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Generalization Gap')
    ax.set_title('Overfitting Monitor (Train Loss - Val Loss)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # Plot 6: Summary Statistics Table
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_data = []
    for model_name in models:
        try:
            history = load_history(model_name)
            best_val_loss = min(history['val_loss'])
            best_epoch = history['val_loss'].index(best_val_loss)
            final_lr = history['learning_rate'][-1]
            total_epochs = len(history['epoch'])
            
            summary_data.append([
                model_name.upper(),
                f"{best_val_loss:.4f}",
                f"{best_epoch + 1}",
                f"{total_epochs}",
                f"{final_lr:.2e}"
            ])
        except:
            continue
    
    if summary_data:
        table = ax.table(
            cellText=summary_data,
            colLabels=['Model', 'Best Val Loss', 'Best Epoch', 'Total Epochs', 'Final LR'],
            cellLoc='center',
            loc='center',
            colWidths=[0.2, 0.2, 0.15, 0.15, 0.15]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(5):
            table[(0, i)].set_facecolor('#4ECDC4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(summary_data) + 1):
            for j in range(5):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F0F0F0')
    
    ax.set_title('Training Summary', fontweight='bold', fontsize=12, pad=20)
    
    plt.tight_layout()
    plt.savefig('results/training_dynamics.png', dpi=300, bbox_inches='tight')
    print("✓ Saved training_dynamics.png")
    plt.show()

def plot_individual_model(model_name):
    """Create detailed plot for individual model"""
    
    history = load_history(model_name)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{model_name.upper()} - Detailed Training Analysis', 
                 fontsize=14, fontweight='bold')
    
    epochs = history['epoch']
    
    # Loss curves with min/max markers
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], label='Train Loss', 
           color='#FF6B6B', linewidth=2.5, marker='o', markersize=3)
    ax.plot(epochs, history['val_loss'], label='Val Loss', 
           color='#4ECDC4', linewidth=2.5, marker='s', markersize=3)
    
    # Mark best epoch
    best_idx = history['val_loss'].index(min(history['val_loss']))
    ax.scatter(best_idx, history['val_loss'][best_idx], 
              color='green', s=200, marker='*', zorder=5, 
              label=f'Best (Epoch {best_idx+1})')
    
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Loss Curves', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Learning rate with annotations
    ax = axes[0, 1]
    ax.plot(epochs, history['learning_rate'], 
           color='#45B7D1', linewidth=2.5, marker='D', markersize=4)
    
    # Annotate LR reductions
    lr_values = history['learning_rate']
    for i in range(1, len(lr_values)):
        if lr_values[i] < lr_values[i-1]:
            ax.annotate('LR Reduced', xy=(i, lr_values[i]), 
                       xytext=(i, lr_values[i]*3),
                       arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                       fontsize=8, color='red')
    
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Learning Rate', fontsize=11)
    ax.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Gradient norm stability
    ax = axes[1, 0]
    grad_norms = history['grad_norm']
    ax.plot(epochs, grad_norms, color='#FFA07A', linewidth=2.5, alpha=0.7)
    ax.fill_between(epochs, grad_norms, alpha=0.3, color='#FFA07A')
    
    mean_grad = np.mean(grad_norms)
    ax.axhline(y=mean_grad, color='red', linestyle='--', linewidth=2, 
              label=f'Mean: {mean_grad:.2f}')
    
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Gradient Norm', fontsize=11)
    ax.set_title('Training Stability (Gradient Norms)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Training metrics summary
    ax = axes[1, 1]
    ax.axis('off')
    
    metrics = [
        ['Metric', 'Value'],
        ['Best Val Loss', f"{min(history['val_loss']):.4f}"],
        ['Final Train Loss', f"{history['train_loss'][-1]:.4f}"],
        ['Best Epoch', f"{best_idx + 1}/{len(epochs)}"],
        ['Initial LR', f"{history['learning_rate'][0]:.1e}"],
        ['Final LR', f"{history['learning_rate'][-1]:.1e}"],
        ['Avg Grad Norm', f"{np.mean(grad_norms):.2f}"],
        ['Total Epochs', f"{len(epochs)}"],
    ]
    
    table = ax.table(cellText=metrics, cellLoc='left', loc='center',
                    colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style
    for i in range(len(metrics)):
        for j in range(2):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#4ECDC4')
                cell.set_text_props(weight='bold', color='white')
            else:
                if i % 2 == 0:
                    cell.set_facecolor('#F8F8F8')
    
    ax.set_title('Training Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f'results/{model_name}_detailed.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved {model_name}_detailed.png")
    plt.show()

def create_comparison_table():
    """Create a comparison table of all models"""
    
    models = ['resnet50', 'efficientnet']
    comparison_data = []
    
    for model_name in models:
        try:
            history = load_history(model_name)
            
            comparison_data.append({
                'Model': model_name.upper(),
                'Best Val Loss': min(history['val_loss']),
                'Final Train Loss': history['train_loss'][-1],
                'Generalization Gap': history['train_loss'][-1] - min(history['val_loss']),
                'Training Epochs': len(history['epoch']),
                'LR Reductions': sum(1 for i in range(1, len(history['learning_rate'])) 
                                    if history['learning_rate'][i] < history['learning_rate'][i-1])
            })
        except:
            continue
    
    if not comparison_data:
        print("No model histories found!")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    
    # Prepare table data
    headers = list(comparison_data[0].keys())
    rows = []
    for data in comparison_data:
        row = []
        for key in headers:
            val = data[key]
            if isinstance(val, float):
                row.append(f"{val:.4f}")
            else:
                row.append(str(val))
        rows.append(row)
    
    table = ax.table(cellText=rows, colLabels=headers, 
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3)
    
    # Style header
    for i, header in enumerate(headers):
        cell = table[(0, i)]
        cell.set_facecolor('#4ECDC4')
        cell.set_text_props(weight='bold', color='white', fontsize=12)
    
    # Alternate row colors
    for i in range(1, len(rows) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')
    
    plt.title('Model Comparison Summary', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved model_comparison.png")
    plt.show()

if __name__ == '__main__':
    print("Creating training visualizations...")
    
    # Create results directory
    Path('results').mkdir(exist_ok=True)
    
    # Check which models have history files
    available_models = []
    for model in ['resnet50', 'efficientnet']:
        if Path(f'results/{model}_history.json').exists():
            available_models.append(model)
    
    if not available_models:
        print("❌ No training history found! Train models first.")
        exit(1)
    
    print(f"Found history for: {', '.join(available_models)}\n")
    
    # Create all plots
    print("1. Creating comprehensive comparison plot...")
    plot_all_metrics(available_models)
    
    print("\n2. Creating detailed individual plots...")
    for model in available_models:
        plot_individual_model(model)
    
    print("\n3. Creating comparison table...")
    create_comparison_table()
    
    print("\n✅ All plots created in results/ directory!")
    print("\nGenerated files:")
    print("  - results/training_dynamics.png")
    for model in available_models:
        print(f"  - results/{model}_detailed.png")
    print("  - results/model_comparison.png")