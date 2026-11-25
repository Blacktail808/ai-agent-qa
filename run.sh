#!/usr/bin/env bash
# Быстрый запуск API. Предварительно: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
