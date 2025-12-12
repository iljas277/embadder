# PDF RAG Telegram Bot

Простой конвейер: PDF → чанки → эмбеддинги → FAISS → ответы в Telegram.

## Быстрый старт
1. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

2. Подготовьте данные:
   - Создайте папку `data/pdfs` (скрипт создаст автоматически).
   - Скопируйте PDF-файлы в `data/pdfs`.

3. Постройте индекс:
   ```bash
   python ingest.py
   ```
   Скрипт извлекает текст из PDF, режет на чанки (RecursiveCharacterTextSplitter, размер 1000, overlap 200),
   считает эмбеддинги моделью `sentence-transformers/all-MiniLM-L6-v2`, сохраняет FAISS-индекс и метаданные в `data/index/`.

4. Настройте окружение (через `.env` рядом с `bot.py`):
   - `TELEGRAM_TOKEN` — токен Telegram бота.
   - `HF_API_TOKEN` — Hugging Face Access Token (https://huggingface.co/settings/tokens).
   - `HF_MODEL` — модель в HF Router (по умолчанию `deepseek-ai/DeepSeek-V3.2`).
   - `HF_INFERENCE_PROVIDER` — провайдер роутера (по умолчанию `novita`).
   - `HF_CHAT_URL` (опционально) — URL chat completions (по умолчанию `https://router.huggingface.co/v1/chat/completions`).

   Пример `.env`:
   ```env
   TELEGRAM_TOKEN=123456:ABCDEF...
   HF_API_TOKEN=hf_...
   HF_MODEL=deepseek-ai/DeepSeek-V3.2
   HF_INFERENCE_PROVIDER=novita
   ```

5. Запустите бота:
   ```bash
   python bot.py
   ```

6. Использование:
   - Отправьте текстовый запрос боту.
   - Он найдёт топ-3 ближайших чанка в векторной базе и сформирует ответ на основе контекста.

## Кастомизация
- Чанкинг: поменяйте `chunk_size`/`overlap` в `ingest.py`.
- Эмбеддер: замените `MODEL_NAME` в `ingest.py` и `vector_store.py`.
- LLM: меняйте `HF_MODEL`/`HF_INFERENCE_PROVIDER` в `.env`.
