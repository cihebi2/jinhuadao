#!/usr/bin/env bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/ui.py --server.port=8501 --server.address=0.0.0.0
