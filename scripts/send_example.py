"""
Пример запроса к нашему API.

Скрипт читает examples/sample_dialog.json, отправляет его на запущенный
API и печатает ответ.

Запуск (API должен быть запущен в другом терминале):
    python scripts/send_example.py
    python scripts/send_example.py http://localhost:8000/analyze_dialog
"""

import json
import sys
from pathlib import Path

import httpx

url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/analyze_dialog"
payload = json.loads(Path("examples/sample_dialog.json").read_text(encoding="utf-8"))

response = httpx.post(url, json=payload, timeout=120)

print(f"HTTP {response.status_code}")
print(json.dumps(response.json(), ensure_ascii=False, indent=2))
