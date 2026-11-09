#!/bin/bash
# AnomalDF baseline: frozen DINOv2 patch matching against N=3 random templates.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m baseline.anomaldf --templates 3 --output outputs/anomaldf
