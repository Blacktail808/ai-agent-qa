# Цифровой сотрудник для анализа качества диалогового AI-агента

Прототип API-приложения на Python: HTTP-сервис на **FastAPI**, выполняющий оценку
качества диалогов диалогового AI-агента с привлечением **LLM** через
**OpenAI-совместимый API**. Ответы формируются механизмом **структурированного
вывода** (JSON Schema) и проходят обязательную валидацию — API возвращает только
валидный JSON.

Учебный проект, тема 2.

---

## 1. Назначение

Сервис принимает диалог AI-агента (роль `assistant`) с пользователем (роль `user`),
передаёт его LLM с инструкцией оценки по заданным критериям и возвращает
структурированный отчёт: итоговая оценка, вердикт, оценки по критериям, перечень
выявленных проблем, рекомендации, признак обнаруженной галлюцинации.

Соответствие требованиям ТЗ:

| Требование ТЗ | Реализация |
|---|---|
| Python — основной язык | Весь код приложения — Python 3.10+ |
| FastAPI — HTTP API | `app/main.py`: эндпоинты `GET /`, `GET /health`, `POST /analyze_dialog` |
| LLM: локально (Ollama/vLLM) или внешняя (OpenRouter, Yandex Cloud, OpenAI) | Переключение конфигурацией `.env` без изменения кода |
| OpenAI-совместимый API как единый интерфейс | Единый клиент `openai` (AsyncOpenAI), параметры `base_url` / `api_key` / `model` |
| Генерация с заданной температурой | `LLM_TEMPERATURE` в `.env` (по умолчанию 0.2) + поле `temperature` в запросе (0.0–1.0) |
| Валидный JSON через структурированный вывод | `response_format: json_schema` (strict) из схемы Pydantic, резервный режим `json_object`, финальная валидация Pydantic |

## 2. Технологический стек

| Компонент | Назначение |
|---|---|
| Python 3.10+ | Язык реализации |
| FastAPI + Uvicorn | HTTP-сервис, проверка запросов по схемам, автодокументация OpenAPI (`/docs`) |
| openai (SDK, AsyncOpenAI) | Клиент OpenAI-совместимого API |
| Pydantic v2 + pydantic-settings | Модели данных, валидация, чтение конфигурации из `.env` |
| LLM-бэкенд | Локально: Ollama или vLLM; внешне: OpenRouter, Yandex Cloud Foundation Models, OpenAI |

Зависимости зафиксированы в `requirements.txt`.

## 3. Архитектура и обработка запроса

```
Клиент ──HTTP──▶ FastAPI (валидация запроса) ──▶ клиент AsyncOpenAI ──▶ LLM-бэкенд
   ▲                                                                        │
   └────── валидный JSON ◀── валидация ответа по AnalyzeResult (Pydantic) ◀─┘
```

Последовательность обработки `POST /analyze_dialog`:

1. FastAPI валидирует тело запроса по модели `AnalyzeRequest`; при несоответствии —
   `422 Unprocessable Entity`.
2. Формируется промпт: системная инструкция эксперта по контролю качества + сериализованный
   диалог + цель агента (`app/llm.py`).
3. Определяется температура: из поля `temperature` запроса, иначе — из настроек.
4. Запрос к LLM с `response_format = json_schema` (strict): схема генерируется из
   модели `AnalyzeResult`, нормализуется (все поля обязательны, `additionalProperties: false`,
   ссылки `$ref` инлайнятся) и передаётся бэкенду.
5. Если бэкенд отклоняет строгую схему — повторный запрос в режиме `json_object`.
6. Ответ модели валидируется моделью `AnalyzeResult`; при несоответствии — `502`.
7. Валидный отчёт сериализуется в JSON и возвращается клиенту (`response_model=AnalyzeResult`).

## 4. Контракт API

### 4.1. `POST /analyze_dialog`

Тело запроса (`AnalyzeRequest`):

| Поле | Тип | Обязательное | Описание |
|---|---|---|---|
| `dialog` | array of `{role, content}` | да (мин. 2) | Диалог; `role` ∈ {`user`, `assistant`, `system`} |
| `agent_goal` | string \| null | нет | Ожидаемая цель агента |
| `temperature` | number (0.0–1.0) \| null | нет | Температура генерации для запроса |

Ответ (`AnalyzeResult`, HTTP 200):

| Поле | Тип | Ограничения |
|---|---|---|
| `overall_score` | integer | 0–100 |
| `verdict` | string | enum: `excellent`, `good`, `needs_improvement`, `poor` |
| `criteria` | array | 3–8 элементов `{name: string, score: int 0–10, comment: string}` |
| `issues` | array of string | может быть пустым |
| `recommendations` | array of string | минимум 1 элемент |
| `hallucination_detected` | boolean | — |

Критерии оценки задаются системным промптом: релевантность, вежливость и стиль,
достижение цели, фактичность, безопасность и этика.

### 4.2. Служебные эндпоинты

- `GET /` — информация о сервисе (модель, адрес LLM, температура по умолчанию).
- `GET /health` — проверка работоспособности.
- `GET /docs` — интерактивная OpenAPI-документация (Swagger UI).

### 4.3. Коды ошибок

| Код | Случай |
|---|---|
| 422 | Тело запроса не соответствует `AnalyzeRequest` |
| 502 | LLM-бэкенд недоступен / отклонил запрос / вернул данные вне схемы |

## 5. Конфигурация

Настройки читаются из `.env` (образец — `.env.example`):

| Переменная | По умолчанию | Описание |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:11434/v1` | Базовый URL OpenAI-совместимого API |
| `LLM_API_KEY` | `not-needed` | Ключ доступа (для Ollama/vLLM не требуется) |
| `LLM_MODEL` | `qwen3:8b` | Имя модели |
| `LLM_TEMPERATURE` | `0.2` | Температура по умолчанию |
| `LLM_TIMEOUT_SECONDS` | `60` | Таймаут запроса к модели |

Пресеты бэкендов:

| Бэкенд | `LLM_BASE_URL` | Пример `LLM_MODEL` |
|---|---|---|
| Ollama (локально) | `http://localhost:11434/v1` | `qwen3:8b` |
| vLLM (локально) | `http://localhost:8000/v1` | имя деплоируемой модели |
| OpenRouter | `https://openrouter.ai/api/v1` | `qwen/qwen-2.5-7b-instruct` |
| Yandex Cloud Foundation Models | `https://foundationmodels.api.cloud.yandex.net/v1` | `yandexgpt-lite` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |

## 6. Установка и запуск

```bash
# 1. Зависимости (один раз)
cd ai-agent-qa
python3 -m venv .venv
source .venv/bin/activate                # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Конфигурация
cp .env.example .env                     # Windows: copy .env.example .env

# 3. LLM-бэкенд (вариант Ollama)
ollama pull qwen3:8b
ollama serve

# 4. Запуск сервиса
./run.sh                                 # uvicorn app.main:app --reload --port 8000
```

Сервис доступен на `http://localhost:8000`, документация — на `http://localhost:8000/docs`.

## 7. Пример использования

```bash
curl -X POST http://localhost:8000/analyze_dialog \
  -H 'Content-Type: application/json' \
  -d @examples/sample_dialog.json
```

Пример ответа (HTTP 200; значения формирует модель, пример иллюстрирует формат):

```json
{
  "overall_score": 58,
  "verdict": "needs_improvement",
  "criteria": [
    { "name": "Релевантность", "score": 7, "comment": "На первый вопрос ответил по делу" },
    { "name": "Вежливость и стиль", "score": 8, "comment": "Тон корректный" },
    { "name": "Достижение цели", "score": 5, "comment": "Проблему с интернетом не решил" },
    { "name": "Фактичность", "score": 2, "comment": "Выдумал тариф и настаивал на нём" },
    { "name": "Безопасность и этика", "score": 9, "comment": "Нарушений нет" }
  ],
  "issues": [
    "Агент сообщил тариф, отсутствующий в данных, и проигнорировал сомнение пользователя"
  ],
  "recommendations": [
    "Запретить агенту называть тарифы без обращения к базе данных клиента"
  ],
  "hallucination_detected": true
}
```

Тестовый диалог `examples/sample_dialog.json` содержит намеренную галлюцинацию агента
(вымышленный тариф) — ожидаемое поведение сервиса: `hallucination_detected: true`
и низкая оценка по критерию «Фактичность».

## 8. Гарантия валидности JSON

1. **Структурированный вывод**: бэкенду передаётся JSON Schema, сгенерированная из
   `AnalyzeResult`, в строгом режиме (`strict: true`, все поля обязательны,
   `additionalProperties: false`).
2. **Резервный режим**: при неподдержке строгой схемы бэкендом — запрос в режиме
   `json_object`.
3. **Финальная валидация**: ответ модели проверяется моделью `AnalyzeResult`
   (`model_validate_json`). Несоответствие — HTTP 502; невалидные данные клиенту
   не возвращаются.

## 9. Структура проекта

```
ai-agent-qa/
├── app/
│   ├── main.py            # Инициализация FastAPI, эндпоинты, обработка ошибок
│   ├── llm.py             # Клиент LLM: промпт, structured output, fallback, валидация
│   ├── schemas.py         # Pydantic-модели: AnalyzeRequest, AnalyzeResult
│   └── config.py          # Настройки из .env (pydantic-settings)
├── examples/sample_dialog.json   # Тестовый запрос
├── scripts/send_example.py       # Скрипт отправки тестового запроса
├── .env.example                  # Шаблон конфигурации
├── requirements.txt              # Зависимости
└── run.sh                        # Скрипт запуска
```

## 10. Проверки, выполненные при разработке

- Запуск сервиса; доступность `GET /`, `GET /health`, `GET /docs`.
- Валидация запроса: некорректное тело отклоняется с кодом 422.
- Обработка недоступного LLM-бэкенда: HTTP 502 с описанием причины.
- Корректность генерируемой JSON Schema (strict-нормализация, инлайнинг `$ref`).
- Валидация эталонных ответов модели: корректный принимается, некорректный
  отклоняется с перечислением нарушений схемы.

## 11. Ограничения и направления развития

- Ограничение прототипа: оценка зависит от выбранной модели и не верифицируется
  внешними эталонами; метрики согласованности оценок не измерялись.
- Развитие: пакетный анализ (`/analyze_batch`), хранение отчётов в БД
  (SQLite/PostgreSQL), сравнение версий промптов агента (A/B), аутентификация,
  ограничение частоты запросов, логирование, модульные тесты.
