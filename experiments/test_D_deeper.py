import sys
sys.path.insert(0, 'D:/ANTIGRAVITY/TEST_HERMES/mem4ristor-v2-main/src')

import numpy as np
np.random.seed(42)

from mem4ristor import Mem4Network

print("=== Direct manipulation of D_eff ===")

# Manually set D_eff and see if coupling changes
net = Mem4Network(size=10, heretic_ratio=0.15, seed=42, cold_start=True)
net.model.cfg['noise']['sigma_v'] = 0.15

# Record initial state
for step in range(100):
    net.step(I_stimulus=0.5)

print(f"Initial state: v_mean={net.v.mean():.4f}, u_mean={net.model.u.mean():.4f}")

# Now manually boost D_eff
original_D_eff = net.model.D_eff
print(f"Original D_eff = {original_D_eff}")

# Inject a large constant laplacian_v to see if D_eff has any effect
test_laplacian = np.ones(100) * 0.5  # Constant gradient

# Run with normal coupling (lattice Laplacian)
net.model.cfg['coupling']['D'] = 0.15
for step in range(100):
    net.step(I_stimulus=0.5)

v_after_normal = net.v.copy()
print(f"\nAfter normal coupling (D=0.15): v_mean={v_after_normal.mean():.4f}, v_std={v_after_normal.std():.4f}")

# Now reset and try with D=5.0
net2 = Mem4Network(size=10, heretic_ratio=0.15, seed=42, cold_start=True)
net2.model.cfg['coupling']['D'] = 5.0
net2.model.cfg['noise']['sigma_v'] = 0.15

for step in range(100):
    net2.step(I_stimulus=0.5)

v_after_high_D = net2.v.copy()
print(f"After high D coupling (D=5.0): v_mean={v_after_high_D.mean():.4f}, v_std={v_after_high_D.std():.4f}")

print(f"\nDifference in v_std: {v_after_high_D.std() - v_after_normal.std():.6f}")
print(f"D_eff ratio: {(5.0/np.sqrt(100)) / (0.15/np.sqrt(100))} = {5.0/0.15:.1f}x")

# Check if v values are actually the same
diff = np.abs(v_after_high_D - v_after_normal)
print(f"Max absolute difference in v: {diff.max():.10f}")
print(f"Mean absolute difference in v: {diff.mean():.10f}")

print("\n=== Check if coupling is being applied at all ===")
# Directly inspect the coupling term
net3 = Mem4Network(size=10, heretic_ratio=0.15, seed=42, cold_start=True)
net3.model.cfg['coupling']['D'] = 1.0

# Run a single step and compute the coupling manually
for step in range(1):
    net3.step(I_stimulus=0.5)

# Compute expected I_coup
laplacian_v = -(net3.L @ net3.v)
u_filter = np.tanh(np.pi * (0.5 - net3.model.u)) + 0.01
D_eff = net3.model.cfg['coupling']['D'] / np.sqrt(net3.N)
expected_I_coup = D_eff * u_filter * laplacian_v

print(f"D_eff = {D_eff:.4f}")
print(f"u_filter: mean={u_filter.mean():.4f}, std={u_filter.std():.4f}")
print(f"laplacian_v: mean={laplacian_v.mean():.6f}, std={laplacian_v.std():.6f}")
print(f"Expected I_coup: mean={expected_I_coup.mean():.6f}, std={expected_I_coup.std():.6f}")
print(f"Expected I_coup range: [{expected_I_coup.min():.6f}, {expected_I_coup.max():.6f}]")

# Now check what happens with D=0.0 (should give zero I_coup if coupling works)
net4 = Mem4Network(size=10, heretic_ratio=0.15, seed=42, cold_start=True)
net4.model.cfg['coupling']['D'] = 0.0
net4.model.cfg['noise']['sigma_v'] = 0.15

for step in range(100):
    net4.step(I_stimulus=0.5)

H_D0 = net4.calculate_entropy()
print(f"\nH_cont with D=0: {H_D0:.4f}")
print(f"H_cont with D=1.0 (same seed): {net3.calculate_entropy():.4f}")