#!/bin/bash
# CAFÉ VIRTUEL PRE-COMMIT GUARDIAN (v2.9)
# NO-BYPASS MODE: This script enforces evidence of failure.

echo "=== CAFÉ VIRTUEL PRE-COMMIT GUARDIAN ==="

# 1. Failure Evidence Check
FAILURE_COUNT=$(ls failures/*.log 2>/dev/null | wc -l)
if [ "$FAILURE_COUNT" -lt 5 ]; then
    echo "❌ FAIL: Integrity requires evidence of failure. Log at least 5 failed runs in /failures/."
    exit 1
fi

# 2. Commit Trace Check
# Ensure new failures are committed with the code
STAGED_FAILURES=$(git diff --cached --name-only | grep "failures/" | wc -l)
if [ "$STAGED_FAILURES" -eq 0 ]; then
    echo "❌ FAIL: You must commit new failure logs alongside your fixes."
    exit 1
fi

# 3. Adversarial Audit Simulation
echo "Running Adversarial Audit (Protocol)..."
export PYTHONPATH="src"
python -m pytest tests/test_kernel.py
if [ $? -ne 0 ]; then
    echo "❌ FAIL: Adversarial unit tests failed."
    exit 1
fi

# 4. Invariant Validation (SNR)
python -c "
import numpy as np
from mem4ristor.core import Mem4ristorV2
m = Mem4ristorV2(seed=42)
m._initialize_params(N=100)
snr = np.abs(m.D_eff * -1.0) / m.cfg['noise']['sigma_v']
assert snr > 2.0, f'SNR too low: {snr}'
"
if [ $? -ne 0 ]; then
    echo "❌ FAIL: Claim Invalidation (SNR < 2.0)."
    exit 1
fi

echo "✅ AUDIT PASSED. Radical integrity verified."
exit 0
