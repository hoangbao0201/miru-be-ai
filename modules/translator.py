"""Translation service that supports multiple providers."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Callable, Dict, List

import requests
from groq import Groq
from openai import OpenAI

from .core.config import TranslatorSettings
from .utils.textblock import TextBlock


class TranslationError(RuntimeError):
    """Raised when translation providers fail."""


def _safe_json_extract(response: str) -> str:
    """Extract JSON payload from provider response."""
    match = re.search(r"\{[\s\S]*\}", response)
    if not match:
        raise TranslationError("Không tìm thấy JSON trong phản hồi của model.")
    return match.group(0)


def _blocks_to_json(blk_list: List[TextBlock]) -> str:
    """Serialize block text into json for prompting."""
    raw = {f"block_{idx}": blk.text for idx, blk in enumerate(blk_list)}
    return json.dumps(raw, ensure_ascii=False, indent=4)


def _update_blocks_from_json(blk_list: List[TextBlock], json_payload: str) -> None:
    """Mutate text blocks with translated strings."""
    translation_dict = json.loads(json_payload)
    for idx, blk in enumerate(blk_list):
        key = f"block_{idx}"
        blk.translation = translation_dict.get(key, blk.text)


class TranslationService:
    """High-level wrapper that hides provider-specific details."""

    def __init__(self, settings: TranslatorSettings) -> None:
        self.settings = settings
        self.provider = settings.provider.lower()
        self._providers: Dict[str, Callable[[str, str], str]] = {
            "openai": self._translate_with_openai,
            "groq": self._translate_with_groq,
            "deepseek": self._translate_with_deepseek,
            "gemini": self._translate_with_gemini,
            "grok": self._translate_with_grok,
        }

    def translate_blocks(
        self, source_lang: str, target_lang: str, blk_list: List[TextBlock]
    ) -> List[TextBlock]:
        """Translate blk_list in place and return it for chaining."""
        user_prompt, system_prompt = self._build_prompts(
            source_lang, target_lang, blk_list
        )
        translator = self._providers.get(self.provider)
        if translator is None:
            raise TranslationError(f"Provider '{self.provider}' chưa được hỗ trợ.")
        raw_response = translator(user_prompt, system_prompt)
        json_payload = _safe_json_extract(raw_response)
        _update_blocks_from_json(blk_list, json_payload)
        return blk_list

    def _build_prompts(
        self, source_lang: str, target_lang: str, blk_list: List[TextBlock]
    ) -> tuple[str, str]:
        """Create tailored prompts per language."""
        entire_raw_text = _blocks_to_json(blk_list)
        if source_lang == "Chinese":
            user_prompt = (
                "Dịch đoạn văn sau một cách tự nhiên, giữ đúng phong cách võ hiệp, "
                f"trang nhã và đầy khí phách:\n{entire_raw_text}"
            )
            system_prompt = (
                f"You are an expert translator who translates {source_lang} to "
                f"{target_lang}. You must return VALID JSON and only change the text "
                "values. Không dịch khóa JSON và không giải thích thêm."
            )
        elif source_lang == "Korean":
            user_prompt = (
                "Make the translation sound as natural as possible.\nTranslate this\n"
                f"{entire_raw_text}"
            )
            system_prompt = (
                f"You are an expert translator who translates {source_lang} to "
                f"{target_lang}. Romanize all proper nouns using the Revised "
                "Romanization (RR) rules. Translate honorifics into the appropriate "
                f"{target_lang} equivalent. Output JSON only."
            )
        else:
            user_prompt = (
                "Make the translation sound as natural as possible.\nTranslate this\n"
                f"{entire_raw_text}"
            )
            system_prompt = (
                f"You are an expert translator who translates {source_lang} to "
                f"{target_lang}. This is comic OCR text: fix typos when confident, "
                f"never translate JSON keys, never output explanations."
            )
        return user_prompt, system_prompt

    def _translate_with_openai(self, user_prompt: str, system_prompt: str) -> str:
        """Translate using OpenAI chat completion."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise TranslationError("OPENAI_API_KEY chưa được cấu hình.")
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=self.settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content

    def _translate_with_groq(self, user_prompt: str, system_prompt: str) -> str:
        """Translate using Groq (Llama3)."""
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise TranslationError("GROQ_API_KEY chưa được cấu hình.")
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=self.settings.groq_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content

    def _translate_with_deepseek(self, user_prompt: str, system_prompt: str) -> str:
        """Translate using Deepseek REST API."""
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise TranslationError("DEEPSEEK_API_KEY chưa được cấu hình.")
        messages = [{"role": "user", "content": user_prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        response = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": "deepseek-chat", "messages": messages},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        return payload["choices"][0]["message"]["content"]

    def _translate_with_gemini(self, user_prompt: str, system_prompt: str) -> str:
        """Translate using Gemini REST API."""
        api_key = "AIzaSyA7sc5zuPW_ILHHzxk37pDbs0Grqv31YiM"
        # os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise TranslationError("GEMINI_API_KEY chưa được cấu hình.")
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.settings.gemini_model}:generateContent"
        )
        body = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": user_prompt}]}],
        }
        response = requests.post(
            url,
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            json=body,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    def _translate_with_grok(self, user_prompt: str, system_prompt: str) -> str:
        """Translate using xAI Grok API."""
        api_key = os.environ.get("XAI_API_KEY")
        if not api_key:
            raise TranslationError("XAI_API_KEY chưa được cấu hình.")
        client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        response = client.chat.completions.create(
            model=self.settings.grok_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=2000,
        )
        return response.choices[0].message.content
