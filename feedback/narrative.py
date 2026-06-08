# Copyright (c) 2026 Callisto Tech — see LICENSE
"""
Generates a narrative summary of all user feedback via Groq.
Called on each render of the feedback panel — not cached, always fresh.
"""

from __future__ import annotations

import logging
import os

from groq import Groq

from feedback.store import load_all

logger = logging.getLogger(__name__)

GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """\
You are an analyst for Callisto Tech, a used car valuation platform.
You receive verbatim user feedback submitted through the app.
Write a concise, professional narrative (3–5 sentences) that synthesises
the themes, sentiments, and any recurring concerns across all comments.
Be objective and specific. Do not use bullet points or headers.
Do not mention that you are an AI or reference the prompt."""


def generate(groq_api_key: str | None = None) -> str:
    entries = load_all()
    if not entries:
        return ""

    api_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    client = Groq(api_key=api_key)

    comments_block = "\n".join(
        f"[{e['ts'][:10]}] {e['comment']}" for e in entries
    )
    user_msg = f"User feedback ({len(entries)} entries):\n\n{comments_block}"

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.4,
        max_tokens=300,
    )

    narrative = response.choices[0].message.content.strip()
    logger.info("Narrative generated from %d feedback entries", len(entries))
    return narrative
