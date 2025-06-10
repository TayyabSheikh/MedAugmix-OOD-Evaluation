import matplotlib.pyplot as plt
import numpy as np
import os

# Data extracted from the corrected report
# Accuracies for 'Best ID Val Acc' checkpoint
resnet50_best = {
    'Methods': ['ERM', 'HypO', 'HypO+MedC'],
    'ID Val Acc': [0.9163, 0.9220, 0.9292],
    'OOD Gen Acc': [0.8021, 0.8232, 0.8440]
}

densenet121_best = {
    'Methods': ['ERM', 'HypO', 'HypO+MedC'],
    'ID Val Acc': [0.9169, 0.9351, 0.9391],
    'OOD Gen Acc': [0.8309, 0.8005, 0.8474]
}

# Accuracies for 'Last Epoch' checkpoint
resnet50_last = {
    'Methods': ['ERM', 'HypO', 'HypO+MedC'],
    'ID Val Acc': [0.9118, 0.9008, 0.9183],
    'OOD Gen Acc': [0.8176, 0.7943, 0.7434]
}

densenet121_last = {
    'Methods': ['ERM', 'HypO', 'HypO+MedC'],
    'ID Val Acc': [0.9113, 0.9266, 0.9276],
    'OOD Gen Acc': [0.8450, 0.8407, 0.8198]
}


output_dir = '../visualizations' # Relative path from hypo_impl/ to visualizations/
os.makedirs(output_dir, exist_ok=True)

def create_comparison_plot(best_data, last_data, model_name, filename):
    """Generates and saves a grouped bar chart comparing accuracies."""
    methods = best_data['Methods']
    id_acc_best = best_data['ID Val Acc']
    ood_acc_best = best_data['OOD Gen Acc']
    id_acc_last = last_data['ID Val Acc']
    ood_acc_last = last_data['OOD Gen Acc']

    x = np.arange(len(methods))  # the label locations
    width = 0.2  # the width of the bars, reduced to fit 4 bars per group

    fig, ax = plt.subplots(figsize=(14, 7)) # Wider figure

    # Positions for the bars
    pos1 = x - 1.5 * width
    pos2 = x - 0.5 * width
    pos3 = x + 0.5 * width
    pos4 = x + 1.5 * width

    rects1 = ax.bar(pos1, id_acc_best, width, label='ID Val Acc (Best Ckpt)', color='deepskyblue')
    rects2 = ax.bar(pos2, ood_acc_best, width, label='OOD Gen Acc (Best Ckpt)', color='lightcoral')
    rects3 = ax.bar(pos3, id_acc_last, width, label='ID Val Acc (Last Epoch)', color='dodgerblue')
    rects4 = ax.bar(pos4, ood_acc_last, width, label='OOD Gen Acc (Last Epoch)', color='indianred')

    # Add some text for labels, title and axes ticks
    ax.set_ylabel('Accuracy')
    ax.set_title(f'{model_name}: Accuracy Comparison (Best ID Val vs Last Epoch Checkpoints)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(loc='lower right') # Adjust legend location

    # Set y-axis limits for better visualization
    all_acc = id_acc_best + ood_acc_best + id_acc_last + ood_acc_last
    min_acc = min(all_acc) if all_acc else 0
    max_acc = max(all_acc) if all_acc else 1
    ax.set_ylim([max(0, min_acc - 0.05), min(1.0, max_acc + 0.05)]) # Adjust ylim based on data range

    # Add value labels on top of bars
    ax.bar_label(rects1, padding=3, fmt='%.4f', rotation=45, size=8)
    ax.bar_label(rects2, padding=3, fmt='%.4f', rotation=45, size=8)
    ax.bar_label(rects3, padding=3, fmt='%.4f', rotation=45, size=8)
    ax.bar_label(rects4, padding=3, fmt='%.4f', rotation=45, size=8)

    fig.tight_layout()
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    print(f"Plot saved to {filepath}")
    plt.close(fig) # Close the figure to free memory

# Create plots
create_comparison_plot(resnet50_best, resnet50_last, 'ResNet50', 'resnet50_accuracy_comparison_best_vs_last.png')
create_comparison_plot(densenet121_best, densenet121_last, 'DenseNet121', 'densenet121_accuracy_comparison_best_vs_last.png')

print("Accuracy comparison plots generated.")
