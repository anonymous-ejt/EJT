import json
import os
import unicodedata

from openai import OpenAI


def create_openai_client():
    api_key = os.getenv("GPT_API_KEY") or os.getenv("OPENAI_API_KEY")

    return OpenAI(api_key=api_key)


def clean_text(text) -> str:
    if text is None:
        return ""

    text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.encode("utf-8", "ignore").decode("utf-8")

    return text.strip()


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)

    if parent:
        os.makedirs(parent, exist_ok=True)


def safe_json_loads(text):
    text = clean_text(text)

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]

        try:
            return json.loads(candidate)
        except Exception:
            pass

    return None
