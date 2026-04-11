import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_ccc_plot():
    csv_path = "results/data/empirical_validation_results.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
    
    df = pd.read_csv(csv_path)
    
    scenarios = df['Scenario'].tolist()
    real_yes = df['Real Yes %'].tolist()
    pred_yes = df['Pred Yes %'].tolist()
    entropy = df['Entropy'].tolist()
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    rects1 = ax1.bar(x - width/2, real_yes, width, label='Real Yes % (CCC)', color='#1f77b4', alpha=0.7)
    rects2 = ax1.bar(x + width/2, pred_yes, width, label='Pred Yes % (Mem4)', color='#ff7f0e', alpha=0.7)
    
    ax1.set_ylabel('Percentage (%)')
    ax1.set_title('Mem4ristor Empirical Validation: CCC France (2020)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, rotation=15, ha='right')
    ax1.legend(loc='upper left')
    
    ax2 = ax1.twinx()
    ax2.plot(x, entropy, color='red', marker='o', linestyle='--', linewidth=2, label='Entropy (H)')
    ax2.set_ylabel('Entropy (Shannon)')
    ax2.set_ylim(0, 2.5)
    ax2.legend(loc='upper right')
    
    fig.tight_layout()
    
    plot_path = "results/plots/ccc_validation_summary.png"
    plt.savefig(plot_path, dpi=300)
    print(f"✅ Plot saved to {plot_path}")

if __name__ == "__main__":
    generate_ccc_plot()
