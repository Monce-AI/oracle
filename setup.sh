#!/bin/bash
python -m venv .venv
source .venv/bin/activate
pip install -e . pandas
echo "Ready. Run: source .venv/bin/activate && python example.py"
