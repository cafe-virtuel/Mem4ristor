# scaling_metrics.py
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('figures', exist_ok=True)

# Données
network_sizes = [1, 16, 100, 625]
state_entropy = [0.0, 0.951, 0.621, 0.384]
mean_doubt = [0.048, 0.051, 0.049, 0.052]
distinct_states = [1, 3, 4, 4]

# Figure
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# A - Entropy
axes[0,0].plot(network_sizes[1:], state_entropy[1:], 's-', linewidth=2.5, color='#2E86AB')
axes[0,0].axhline(y=0.38, color='gray', linestyle=':', label='Min = 0.38')
axes[0,0].set_xscale('log')
axes[0,0].set_xlabel('Network Size')
axes[0,0].set_ylabel('Shannon Entropy')
axes[0,0].set_title('A) Cognitive Diversity')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# B - Doubt
axes[0,1].plot(network_sizes, mean_doubt, '^-', linewidth=2.5, color='#A23B72')
axes[0,1].axhline(y=0.05, color='gray', linestyle=':', label='Baseline')
axes[0,1].set_xscale('log')
axes[0,1].set_xlabel('Network Size')
axes[0,1].set_ylabel('Mean Doubt (u)')
axes[0,1].set_title('B) Epistemic Uncertainty')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# C - Oracle fraction (toujours 0)
axes[1,0].plot(network_sizes, [0,0,0,0], 'o-', linewidth=2.5, color='#C73E1D', label='v2.0')
axes[1,0].axhline(y=0.86, color='gray', linestyle='--', label='v1.0 (86%)')
axes[1,0].set_xscale('log')
axes[1,0].set_xlabel('Network Size')
axes[1,0].set_ylabel('Oracle Fraction')
axes[1,0].set_title('C) No Oracle Collapse')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# D - Distinct states
bars = axes[1,1].bar(range(4), distinct_states, color=['#F18F01', '#6B2737', '#2E86AB', '#A23B72'])
axes[1,1].set_xticks(range(4))
axes[1,1].set_xticklabels(['1', '16', '100', '625'])
axes[1,1].set_ylim(0, 5)
axes[1,1].set_xlabel('Network Size')
axes[1,1].set_ylabel('Distinct States')
axes[1,1].set_title('D) State Richness')

# Annotations
for i, v in enumerate(distinct_states):
    axes[1,1].text(i, v + 0.1, str(v), ha='center', fontweight='bold')

plt.suptitle('Mem4ristor v2.0: Scaling Performance', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('figures/scaling_metrics.pdf', dpi=300, bbox_inches='tight')
print("✅ Scaling metrics sauvegardées")
plt.show()