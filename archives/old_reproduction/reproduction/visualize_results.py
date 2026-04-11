import json
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    results_path = "projects/mem4ristor-v2/benchmarks/data/benchmark_results_v2_1.json"
    output_dir = "projects/mem4ristor-v2/docs/figures"
    os.makedirs(output_dir, exist_ok=True)

    with open(results_path, "r") as f:
        results = json.load(f)

    models = [res["model"] for res in results]
    mean_dd = [res["mean_dd"] for res in results]
    std_dd = [res["std_dd"] for res in results]
    ci95_low = [res["ci95_low"] for res in results]
    ci95_high = [res["ci95_high"] for res in results]
    
    # 1. DD Score Bar Chart with CI95
    plt.figure(figsize=(10, 6))
    colors = ['#4F46E5', '#10B981', '#F59E0B', '#EF4444']
    y_err = [np.array(mean_dd) - np.array(ci95_low), np.array(ci95_high) - np.array(mean_dd)]
    
    bars = plt.bar(models, mean_dd, yerr=y_err, capsize=10, color=colors, alpha=0.8, edgecolor='black')
    
    plt.title("Deliberative Diversity (DD) Score comparison\n(Agora Scientific Standard)", fontsize=14, fontweight='bold')
    plt.ylabel("DD Score", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dd_score_comparison.png"), dpi=300)
    print(f"✅ Saved DD comparison to {output_dir}/dd_score_comparison.png")

    # 2. Entropy vs Bias Alignment Scatter Plot
    plt.figure(figsize=(10, 6))
    h_vals = [res["h_final"] for res in results]
    b_vals = [res["bias_align"] for res in results]
    
    for i, model in enumerate(models):
        plt.scatter(b_vals[i], h_vals[i], color=colors[i % len(colors)], s=200, label=model, edgecolors='black', alpha=0.9)
        plt.annotate(model, (b_vals[i], h_vals[i]), xytext=(10, 10), textcoords='offset points', fontsize=10)

    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    
    plt.xlabel("Bias Alignment |B_final| (Lower is better)", fontsize=12)
    plt.ylabel("Final Entropy H_final (Higher is more diverse)", fontsize=12)
    plt.title("Cognitive Diversity vs. Bias Independence", fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "diversity_vs_bias.png"), dpi=300)
    print(f"✅ Saved Diversity vs Bias to {output_dir}/diversity_vs_bias.png")

if __name__ == "__main__":
    main()
