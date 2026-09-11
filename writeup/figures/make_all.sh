#!/bin/sh
# Regenerate every figure from results/. Run from the repo root or anywhere.
set -e
cd "$(dirname "$0")"
for f in fig1_stories.py fig2_graph.py fig3_calibration.py fig4_huginn.py fig5_text.py fig6_checklist.py; do
  [ -f "$f" ] && ../../.venv/bin/python "$f"
done
