#!/usr/bin/env bash
# Run all EigenDialectos experiments sequentially
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "EigenDialectos: Running All Experiments"
echo "======================================="
echo ""

EXPERIMENTS=(
    "exp1_spectral_map"
    "exp2_full_generation"
    "exp3_dialectal_gradient"
    "exp4_impossible_dialects"
    "exp5_archaeology"
    "exp6_evolution"
    "exp7_zeroshot"
)

PASSED=0
FAILED=0

for exp in "${EXPERIMENTS[@]}"; do
    echo "--- Running: ${exp} ---"
    if python -m eigendialectos.cli.main experiment run "$exp" 2>&1; then
        echo "  PASSED: ${exp}"
        PASSED=$((PASSED + 1))
    else
        echo "  FAILED: ${exp}"
        FAILED=$((FAILED + 1))
    fi
    echo ""
done

echo "======================================="
echo "Results: ${PASSED} passed, ${FAILED} failed out of ${#EXPERIMENTS[@]}"

if [ "$FAILED" -gt 0 ]; then
    exit 1
fi
