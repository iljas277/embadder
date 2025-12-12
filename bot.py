import logging
import os
import sys
import time
from typing import List, Tuple

import requests
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest

from vector_store import VectorStore

load_dotenv()  # подхватит .env рядом с bot.py

# --- logging ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_TEXT = os.getenv("LOG_TEXT", "0") == "1"  # 1 = логировать превью входного текста/ответа

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("ht_tgbot")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)


def _preview(text: str, n: int = 200) -> str:
    t = (text or "").replace("\n", "\\n")
    return t[:n] + ("…" if len(t) > n else "")


# --- env ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_CHAT_URL = os.getenv("HF_CHAT_URL", "https://router.huggingface.co/v1/chat/completions")
HF_MODEL = os.getenv("HF_MODEL", "deepseek-ai/DeepSeek-V3.2")
HF_INFERENCE_PROVIDER = os.getenv("HF_INFERENCE_PROVIDER", "novita")

# --- runtime ---
START_TS = time.monotonic()

# RAG
vs = VectorStore()


def build_prompt(query: str, contexts: List[str]) -> str:
    """Формируем подсказку для LLM с контекстом чанков."""
    ctx_block = "\n\n---\n\n".join(contexts) if contexts else "(контекст не найден)"
    return (
        "Ты — помощник. Отвечай КОРОТКО и ПО ДЕЛУ, опираясь ТОЛЬКО на контекст ниже. "
        "Если ответа в контексте нет — так и скажи.\n\n"
        f"КОНТЕКСТ:\n{ctx_block}\n\n"
        f"ВОПРОС: {query}\n"
        "ОТВЕТ:"
    )


def build_requests_session() -> requests.Session:
    """Создаём requests.Session без использования прокси."""
    session = requests.Session()
    session.trust_env = False  # игнорируем HTTP(S)_PROXY из окружения
    session.proxies = {}
    return session


SESSION = build_requests_session()


def call_llm(prompt: str) -> str:
    """Вызываем LLM через Hugging Face Router (OpenAI-compatible chat completions)."""
    if not HF_API_TOKEN:
        logger.error("HF_API_TOKEN is missing")
        return "Не задан HF_API_TOKEN (положите в .env)."

    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json",
    }
    if HF_INFERENCE_PROVIDER:
        headers["X-Inference-Provider"] = HF_INFERENCE_PROVIDER

    payload = {
        "model": HF_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 256,
    }

    t0 = time.monotonic()
    try:
        resp = SESSION.post(HF_CHAT_URL, headers=headers, json=payload, timeout=120)
    except Exception:
        logger.exception("HF Router request failed")
        return "Ошибка сети/таймаут при запросе к LLM."

    dt = time.monotonic() - t0
    logger.info(
        "LLM call done: status=%s time=%.2fs model=%s provider=%s",
        resp.status_code,
        dt,
        HF_MODEL,
        HF_INFERENCE_PROVIDER,
    )

    if resp.status_code >= 400:
        body = (resp.text or "").strip()
        logger.error("LLM error: status=%s body_preview=%s", resp.status_code, _preview(body, 400))
        return f"HF Router ошибка {resp.status_code}: {body}"

    try:
        data = resp.json()
    except Exception:
        logger.error("LLM returned non-JSON: %s", _preview(resp.text, 400))
        return f"HF Router вернул не-JSON: {resp.text[:500]}"

    try:
        out = data["choices"][0]["message"]["content"].strip()
        if LOG_TEXT:
            logger.info("LLM answer preview: %s", _preview(out, 200))
        return out
    except Exception:
        logger.error("LLM response parse failed: %r", data)
        return f"Не удалось распарсить ответ: {data!r}"


def _split_telegram(text: str, limit: int = 4000) -> List[str]:
    text = text or ""
    if len(text) <= limit:
        return [text]
    parts: List[str] = []
    cur = 0
    while cur < len(text):
        parts.append(text[cur : cur + limit])
        cur += limit
    return parts


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Health-check: жив ли бот, видит ли индекс, настроена ли LLM."""
    uptime_s = int(time.monotonic() - START_TS)

    ntotal = None
    meta_len = None
    try:
        ntotal = int(getattr(vs.index, "ntotal", -1))
    except Exception:
        ntotal = None
    try:
        meta_len = len(getattr(vs, "meta", []))
    except Exception:
        meta_len = None

    text = (
        "✅ Bot status: OK\n"
        f"uptime: {uptime_s}s\n"
        f"telegram_token: {'yes' if bool(TELEGRAM_TOKEN) else 'NO'}\n"
        f"hf_token: {'yes' if bool(HF_API_TOKEN) else 'NO'}\n"
        f"hf_model: {HF_MODEL}\n"
        f"hf_provider: {HF_INFERENCE_PROVIDER}\n"
        f"hf_chat_url: {HF_CHAT_URL}\n"
        f"faiss_ntotal: {ntotal}\n"
        f"chunks_meta: {meta_len}\n"
        f"log_level: {LOG_LEVEL}\n"
        f"log_text: {int(LOG_TEXT)}"
    )
    await update.message.reply_text(text)


async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("pong")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    user_text = update.message.text.strip()
    if not user_text:
        return

    chat_id = getattr(update.effective_chat, "id", None)
    user_id = getattr(update.effective_user, "id", None)

    logger.info("message: chat_id=%s user_id=%s len=%s", chat_id, user_id, len(user_text))
    if LOG_TEXT:
        logger.info("message preview: %s", _preview(user_text, 200))

    # RAG
    t0 = time.monotonic()
    results: List[Tuple[str, float]] = vs.search(user_text, top_k=3)
    rag_dt = time.monotonic() - t0
    contexts = [txt for (txt, _score) in results]

    logger.info(
        "RAG: hits=%d time=%.2fs scores=%s",
        len(results),
        rag_dt,
        [round(s, 4) for (_t, s) in results],
    )

    prompt = build_prompt(user_text, contexts)
    answer = call_llm(prompt)

    for part in _split_telegram(answer):
        await update.message.reply_text(part)


def build_request() -> HTTPXRequest:
    return HTTPXRequest(
        connect_timeout=20.0,
        read_timeout=60.0,
        write_timeout=60.0,
        pool_timeout=20.0,
    )


def main():
    if not TELEGRAM_TOKEN:
        raise RuntimeError("Не задан TELEGRAM_TOKEN (положите в .env).")

    logger.info(
        "Starting bot. model=%s provider=%s chat_url=%s log_level=%s",
        HF_MODEL,
        HF_INFERENCE_PROVIDER,
        HF_CHAT_URL,
        LOG_LEVEL,
    )

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).request(build_request()).build()

    # команды
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("ping", cmd_ping))

    # сообщения
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
