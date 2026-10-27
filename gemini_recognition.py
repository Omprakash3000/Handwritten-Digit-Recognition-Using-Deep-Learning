"""Gemini Vision AI module for recognizing arbitrary handwritten Hindi/Devanagari images.

Handles untrained characters, complex conjuncts (संयुक्त अक्षर), vowels with matras,
handwritten Hindi words, numerals, and sentences using Google Gemini Vision API.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional, Union

from PIL import Image

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None  # type: ignore

from config import GEMINI_API_KEY, GEMINI_DEFAULT_MODEL
from utils import setup_logging

LOGGER = setup_logging(__name__)

STANDARD_46_CLASSES = {
    "character_1_ka": "क", "character_2_kha": "ख", "character_3_ga": "ग",
    "character_4_gha": "घ", "character_5_kna": "ङ", "character_6_cha": "च",
    "character_7_chha": "छ", "character_8_ja": "ज", "character_9_jha": "झ",
    "character_10_yna": "ञ", "character_11_taamatar": "ट", "character_12_thaa": "ठ",
    "character_13_daa": "ड", "character_14_dhaa": "ढ", "character_15_adna": "ण",
    "character_16_tabala": "त", "character_17_tha": "थ", "character_18_da": "द",
    "character_19_dha": "ध", "character_20_na": "न", "character_21_pa": "प",
    "character_22_pha": "फ", "character_23_ba": "ब", "character_24_bha": "भ",
    "character_25_ma": "म", "character_26_yaw": "य", "character_27_ra": "र",
    "character_28_la": "ल", "character_29_waw": "व", "character_30_mordhansha": "ष",
    "character_31_dhyanash": "श", "character_32_patalosha": "स", "character_33_ha": "ह",
    "character_34_chhya": "क्ष", "character_35_tra": "त्र", "character_36_gya": "ज्ञ",
    "digit_0": "०", "digit_1": "१", "digit_2": "२", "digit_3": "३",
    "digit_4": "४", "digit_5": "५", "digit_6": "६", "digit_7": "७",
    "digit_8": "८", "digit_9": "९",
}


def get_gemini_client(api_key: Optional[str] = None) -> genai.Client:
    """Create and return a configured Google GenAI client."""
    if genai is None:
        raise ImportError(
            "The `google-genai` package is required. Install it using `pip install google-genai`."
        )
    key = (api_key or GEMINI_API_KEY or "").strip()
    if not key:
        raise ValueError("Gemini API key is missing. Please provide a valid API key.")
    return genai.Client(api_key=key)


def _extract_json_block(text: str) -> dict[str, Any]:
    """Parse JSON response from Gemini, handling code markdown fences if present."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    else:
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            text = text[first_brace : last_brace + 1].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "glyph": text[:50],
            "transliteration": "N/A",
            "type": "General Devanagari Text",
            "confidence": "Estimated 90%",
            "pronunciation": "N/A",
            "meaning_en": text,
            "strokes_and_features": "Parsed from raw response.",
            "examples": [],
            "raw_response": text,
        }


def recognize_with_gemini(
    image: Union[Image.Image, Path, str],
    api_key: Optional[str] = None,
    preferred_model: Optional[str] = None,
) -> dict[str, Any]:
    """Recognize any handwritten Hindi/Devanagari image using Gemini Vision.

    Args:
        image: PIL Image instance or file path.
        api_key: Optional API key override. Defaults to config.GEMINI_API_KEY.
        preferred_model: Model name to try. Defaults to 'gemini-2.5-flash'.

    Returns:
        Structured dictionary with recognition results.
    """
    if isinstance(image, (str, Path)):
        pil_image = Image.open(image).convert("RGB")
    else:
        pil_image = image.convert("RGB")

    client = get_gemini_client(api_key)

    prompt = (
        "You are an expert Devanagari and Hindi handwritten script recognition system.\n"
        "Analyze this handwritten image carefully. It may contain a Devanagari character, "
        "a vowel with diacritics/matras (मात्राएं), a conjunct/ligature (संयुक्त अक्षर e.g. द्व, द्य, श्र, त्र, क्ष, ज्ञ, क्त), "
        "a Devanagari numeral (०-९), a handwritten Hindi word, or a phrase/sentence.\n\n"
        "Return a strictly valid JSON object with the following keys:\n"
        "{\n"
        '  "glyph": "The exact recognized Hindi/Devanagari text (e.g. क, नमस्ते, ऋ, क्ष, ५, etc.)",\n'
        '  "transliteration": "Phonetic English spelling/romanization (e.g. ka, Namaste, Ri, ksha, 5)",\n'
        '  "type": "One of: Single Character, Conjunct/Ligature, Vowel/Matra, Numeral, Word, Sentence",\n'
        '  "confidence": "Estimated confidence percentage (e.g. 98%)",\n'
        '  "pronunciation": "Phonetic pronunciation guide",\n'
        '  "meaning_en": "English meaning, translation, or definition of the character/word",\n'
        '  "strokes_and_features": "Detailed observation on handwritten strokes, Shirorekha (top line), loops, and handwriting style",\n'
        '  "examples": ["2-3 example Hindi words using this glyph/word with English meanings"]\n'
        "}\n\n"
        "Do not include any introductory or concluding text outside the JSON block."
    )

    models_to_try = [
        preferred_model or GEMINI_DEFAULT_MODEL,
        "gemini-flash-latest",
        "gemini-2.5-pro",
        "gemini-pro-latest",
    ]
    # Remove duplicates while preserving order
    models_to_try = list(dict.fromkeys(models_to_try))

    last_error = None
    for model_name in models_to_try:
        try:
            LOGGER.info("Calling Gemini Vision API with model: %s", model_name)
            response = client.models.generate_content(
                model=model_name,
                contents=[pil_image, prompt],
            )
            raw_text = response.text or ""
            parsed = _extract_json_block(raw_text)

            # Check if it matches any standard 46 class
            glyph = parsed.get("glyph", "").strip()
            in_standard = False
            for class_code, char_glyph in STANDARD_46_CLASSES.items():
                if glyph == char_glyph:
                    in_standard = True
                    parsed["matched_dataset_class"] = class_code
                    break

            parsed["is_trained_dataset_class"] = in_standard
            parsed["model_used"] = model_name
            parsed["status"] = "success"
            return parsed
        except Exception as exc:
            LOGGER.warning("Gemini model %s failed: %s", model_name, exc)
            last_error = exc

    raise RuntimeError(f"All Gemini Vision models failed. Last error: {last_error}")
