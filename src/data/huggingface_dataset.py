import json
import re
from typing import Any, Dict

from .base_dataset import BaseDatasetHandler


def _dict_list_to_chat(tokenizer, conv: list[dict[str, Any]]) -> dict[str, str]:
    """Helper function to convert a list of dicts to a chat string."""
    norm = []
    for turn in conv:
        role = (turn.get("role") or turn.get("from") or "").lower()
        if role in {"human", "user"}:
            role = "user"
        elif role in {"assistant", "gpt", "model"}:
            role = "assistant"
        norm.append({"role": role, "content": turn.get("content") or turn.get("value") or ""})

    norm = [t for t in norm if t["content"] and t["content"].strip()]
    if not norm:
        return None

    try:
        return {"text": tokenizer.apply_chat_template(norm, tokenize=False)}
    except Exception:
        joined = "\n".join(f"{t['role'].capitalize()}: {t['content']}" for t in norm)
        return {"text": joined}


class SFTDataset(BaseDatasetHandler):
    """
    Handles chat and instruction-formatted datasets (SFT).
    Inherits all processing logic from BaseDatasetHandler.
    """

    def _process_text_column(self, examples: Dict[str, Any]) -> Dict[str, str]:
        """
        Normalizes various chat and instruction formats into a single 'text' field.
        This is the specialized logic for this handler.
        """
        preferred = self.text_column
        raw = examples.get(preferred)

        if raw is None:
            for alt in (
                "messages",
                "conversation",
                "conversations",
                "prompt_response",
                "text",
                "chosen",
            ):
                if alt in examples:
                    raw = examples[alt]
                    break

        if raw is None:
            q = examples.get("query") or examples.get("prompt")
            a = examples.get("response") or examples.get("answer")
            if q is not None and a is not None:
                return _dict_list_to_chat(
                    self.tokenizer,
                    [{"role": "user", "content": q}, {"role": "assistant", "content": a}],
                )

        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            return _dict_list_to_chat(self.tokenizer, raw) or {"text": ""}

        if isinstance(raw, str):
            if raw.strip().startswith(("{ ", "[")):
                try:
                    obj = json.loads(raw)
                    if isinstance(obj, list):
                        return _dict_list_to_chat(self.tokenizer, obj) or {"text": ""}
                except Exception:
                    pass

            blocks = re.split(r"###\s*|\n(?=\s*(Human|Assistant|User):)", raw.strip())
            conv = []
            for blk in blocks:
                if not isinstance(blk, str):
                    continue
                m = re.match(r"\s*(Human|Assistant|User)\s*:\s*(.*)", blk, flags=re.S)
                if m:
                    role = "user" if m.group(1) in {"Human", "User"} else "assistant"
                    content = m.group(2).strip()
                    if content:
                        conv.append({"role": role, "content": content})
            if conv:
                return _dict_list_to_chat(self.tokenizer, conv) or {"text": ""}

        return {"text": str(raw).strip() if raw is not None else ""}
