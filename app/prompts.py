from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, StrictUndefined

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RenderedPrompt:
    name: str
    version: str
    text: str
    prompt_hash: str
    model: str | None
    temperature: float | None


class PromptRegistry:
    """Versioned YAML prompts with strict variable checking."""

    def __init__(self, directory: str | Path | None = None) -> None:
        self.directory = Path(directory or ROOT / "prompts")
        self.env = Environment(undefined=StrictUndefined, autoescape=False, trim_blocks=True, lstrip_blocks=True)
        self._items: dict[str, dict[str, Any]] = {}
        self.reload()

    def reload(self) -> None:
        self._items.clear()
        for path in sorted(self.directory.glob("*.yaml")):
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            name = data.get("name")
            if not name:
                raise ValueError(f"Prompt缺少name：{path}")
            if name in self._items:
                raise ValueError(f"Prompt名称重复：{name}")
            data["_path"] = str(path)
            self._items[name] = data

    def names(self) -> list[str]:
        return sorted(self._items)

    def render(self, name: str, **values: Any) -> RenderedPrompt:
        item = self._items.get(name)
        if not item:
            raise KeyError(f"Prompt不存在：{name}")
        required = set(item.get("input_variables", []))
        missing = required - set(values)
        if missing:
            raise ValueError(f"Prompt {name} 缺少变量：{', '.join(sorted(missing))}")
        text = self.env.from_string(item["template"]).render(**values).strip()
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return RenderedPrompt(
            name=name,
            version=str(item.get("version", "1.0.0")),
            text=text,
            prompt_hash=digest,
            model=item.get("model"),
            temperature=item.get("temperature"),
        )
