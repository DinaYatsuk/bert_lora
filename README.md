# handler_template

Это темплейт второго по очереди, но не по популярности способа инференса готовых моделей.
Идея состоит в том, что вы предоставляете API (аналог питоновской функции, только для внешних приложений),
и по этому API другие юзеры/приложения могут обращаться к вашей модели.
Рекомендую посмотреть курс по [FastAPI](https://www.youtube.com/watch?v=JqfbfraMjW8&list=PLeLN0qH0-mCVQKZ8-W1LhxDcVlWtTALCS&ab_channel=%D0%90%D1%80%D1%82%D1%91%D0%BC%D0%A8%D1%83%D0%BC%D0%B5%D0%B9%D0%BA%D0%BE)
и видео по организации кода из [этого плейлиста (видео номер 3 про кодстайл)](https://buildin.ai/bade308d-f219-4787-b914-c754121c05f6#d89b8ede-8dbc-40b6-8247-2df5972c7321).

## Review Sentiment API

Сервис определяет тональность отзыва (`negative` / `neutral` / `positive`) моделью RuBERT / RuModernBERT,
дообученной через LoRA (ноутбук `rubert_lora_sentiment.ipynb`).

### Данные для обучения

Модель обучалась на датасете [RuReviews](https://github.com/sismetanin/rureviews):
отзывы на женскую одежду и аксессуары на русском языке, 90 000 отзывов с разметкой тональности
(`negative` / `neutral` / `positive`). Разметка в датасете автоматическая, по оценке отзыва,
поэтому на отзывах из других предметных областей (электроника, еда, услуги) качество может быть ниже.


### Структура проекта

```
handler_template/
├── lib/
│   ├── __init__.py
│   ├── clients/
│   │   └── classification_client.py   # Python-клиент для этого API
│   ├── constants/
│   │   └── settings.py                # Настройки (переменные окружения)
│   ├── handlers/
│   │   └── classify.py                # Ручки: /classify, /classify/batch, /health
│   ├── models/
│   │   └── bert_classifier.py         # Загрузка LoRA-модели и предсказание (технический слой)
│   ├── processors/
│   │   └── text_processor.py          # Предобработка текста
│   ├── utils/
│   │── notebooks/
│   │   └── bert_train.ipync           # Блокнот с обучением модели
│   ├── dto.py                         # Pydantic-модели входных/выходных данных
│   └── main.py                        # FastAPI-приложение
├── requirements.txt
└── README.md
```

### Установка

```bash
pip install -r requirements.txt
```

### Модель

Базовая модель (`deepvk/RuModernBERT-base` и т.д.) и тип головы читаются из `labels.json`,
Базовая модель скачивается с HuggingFace при первом запуске.

### Запуск API сервера

Запускать из корня проекта (папки `bert_lora/`):

```bash
uvicorn main:app --app-dir lib --port 8000
```

После запуска:
- API: http://localhost:8000
- Документация: http://localhost:8000/docs
- Health check: http://localhost:8000/health
- Gradio: http://localhost:8000/ui

### Использование

#### Через клиент (Python)

```python
from lib.clients import ClassificationClient

client = ClassificationClient(base_url="http://localhost:8000")

resp = client.classify("Отличная куртка, швы ровные, запаха нет!")
print(resp.label)         # positive
print(resp.predictions)   # [LabelScore(label='positive', score=0.97), ...]

batch = client.classify_batch(["Ужасное качество", "Нормально за свои деньги"], top_k=2)
```

#### Через HTTP запрос

```bash
curl -X POST "http://localhost:8000/classify" \
     -H "Content-Type: application/json" \
     -d '{"text": "Отличная куртка, швы ровные!", "top_k": 2}'
```

Ответ:

```json
{
  "label": "positive",
  "predictions": [
    {"label": "positive", "score": 0.97},
    {"label": "neutral", "score": 0.02}
  ]
}
```

### Ручки

| Метод | Путь | Описание |
|---|---|---|
| POST | `/classify` | Один текст: `{"text": str}` |
| POST | `/classify/batch` | Несколько текстов: `{"texts": [str]}` |
| GET | `/health` | Статус, список классов, устройство |

Ошибки: `422` - пустой текст, `413` - слишком длинный текст или слишком большой батч,
`503` - модель ещё не загружена.