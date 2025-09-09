# agent_mcp_openrouter.py
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletion
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# =========================
# Config
# =========================

@dataclass
class Settings:
    openrouter_api_key: str
    model: str
    brightdata_token: str
    mcp_url: str
    temperature: float
    max_steps: int
    tool_timeout: float

    @classmethod
    def load(cls) -> "Settings":
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key or not api_key.startswith("sk-or-"):
            raise RuntimeError(
                "OPENROUTER_API_KEY не найден или неверного формата. "
                "Откройте https://openrouter.ai/ → Keys → создайте ключ вида sk-or-..."
            )
        model = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")

        token = os.getenv("BRIGHTDATA_TOKEN")
        if not token:
            raise RuntimeError(
                "BRIGHTDATA_TOKEN не найден. Задайте токен Bright Data в .env или окружении."
            )
        mcp_url = os.getenv("MCP_URL", f"https://mcp.brightdata.com/mcp?token={token}")
        temperature = float(os.getenv("AGENT_TEMPERATURE", "0.2"))
        max_steps = int(os.getenv("AGENT_MAX_STEPS", "4"))
        tool_timeout = float(os.getenv("TOOL_TIMEOUT", "180"))

        return cls(
            openrouter_api_key=api_key,
            model=model,
            brightdata_token=token,
            mcp_url=mcp_url,
            temperature=temperature,
            max_steps=max_steps,
            tool_timeout=tool_timeout,
        )

# =========================
# MCP helpers (короткие сессии)
# =========================

async def mcp_list_tools(mcp_url: str) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Возвращает (OpenAI-совместимые tools, schema_by_name),
    открывая короткую MCP-сессию (устойчиво вне ноутбуков и в них тоже).
    """
    async with streamablehttp_client(mcp_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            desc = await session.list_tools()

    openai_tools: List[Dict[str, Any]] = []
    schemas: Dict[str, Dict[str, Any]] = {}
    for t in desc.tools:
        schema = t.inputSchema or {"type": "object", "properties": {}}
        schemas[t.name] = schema
        openai_tools.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": schema
            }
        })
    return openai_tools, schemas


async def mcp_call(mcp_url: str, tool_name: str, arguments: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    """
    Выполнить вызов MCP-тула через короткую сессию. Возвращает payload вида:
    { "isError": bool, "content": [...parts...] } или объект ошибки.
    """
    try:
        async with streamablehttp_client(mcp_url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                res = await asyncio.wait_for(
                    session.call_tool(name=tool_name, arguments=arguments),
                    timeout=timeout
                )
        return {
            "isError": res.isError,
            "content": [c.model_dump() for c in res.content]
        }
    except Exception as e:
        return {
            "isError": True,
            "why": "mcp_call_exception",
            "message": str(e),
            "traceback": traceback.format_exc(),
            "called_tool": tool_name,
            "called_args": arguments
        }

# =========================
# Аргументы: приведение к схеме
# =========================

def _apply_defaults(args: Dict[str, Any], schema: Dict[str, Any]) -> None:
    props = schema.get("properties", {}) or {}
    for k, v in props.items():
        if k not in args and isinstance(v, dict) and "default" in v:
            args[k] = v["default"]

def _rename_common_aliases(tool_name: str, args: Dict[str, Any], schema: Dict[str, Any]) -> None:
    props = schema.get("properties", {}) or {}
    if tool_name == "search_engine":
        if "query" in props and "query" not in args:
            if "q" in args:
                args["query"] = args.pop("q")
            elif "text" in args:
                args["query"] = args.pop("text")

def _strip_additional(args: Dict[str, Any], schema: Dict[str, Any]) -> None:
    props = set((schema.get("properties") or {}).keys())
    if schema.get("additionalProperties") is False:
        for k in list(args.keys()):
            if k not in props:
                args.pop(k, None)

def _ensure_required(args: Dict[str, Any], schema: Dict[str, Any]) -> Optional[str]:
    required = schema.get("required") or []
    missing = [r for r in required if r not in args]
    if missing:
        return f"Missing required fields: {', '.join(missing)}"
    return None

def normalize_args(tool_name: str, raw_args: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    args = dict(raw_args or {})
    try:
        _rename_common_aliases(tool_name, args, schema)
        _apply_defaults(args, schema)
        _strip_additional(args, schema)
        err = _ensure_required(args, schema)
        return args, err
    except Exception as e:
        return args, f"Normalization error: {e}"

# =========================
# Агент: OpenRouter ↔ MCP
# =========================

class MCPAgent:
    def __init__(self, client: OpenAI, model: str, mcp_url: str, temperature: float = 0.2, max_steps: int = 4, tool_timeout: float = 180.0):
        self.client = client
        self.model = model
        self.mcp_url = mcp_url
        self.temperature = temperature
        self.max_steps = max_steps
        self.tool_timeout = tool_timeout

    async def run(self, user_prompt: str) -> str:
        # 1) Получаем инструменты и схемы (короткая сессия)
        openai_tools, schemas = await mcp_list_tools(self.mcp_url)

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content":
             ("Ты агент с доступом к MCP (Bright Data). "
              "Если нужны свежие данные/поиск/парсинг — используй инструменты. "
              "После вызова инструмента извлекай факты и отвечай кратко, со ссылками. Для поиска всегда используй Google.")},
            {"role": "user", "content": user_prompt},
        ]

        last_text_reply = ""

        for _ in range(self.max_steps):
            # 2) Шаг модели
            resp: ChatCompletion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=openai_tools,
                tool_choice="auto",
                temperature=self.temperature,
            )
            msg = resp.choices[0].message
            if msg.content:
                last_text_reply = msg.content

            tool_calls = msg.tool_calls or []
            if not tool_calls:
                return msg.content or last_text_reply or ""

            # 3) Исполняем каждую запрошенную функцию через MCP (короткая сессия)
            for tc in tool_calls:
                try:
                    raw_args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    raw_args = {}

                schema = schemas.get(tc.function.name, {"type": "object", "properties": {}})
                args, norm_err = normalize_args(tc.function.name, raw_args, schema)

                # echo assistant tool_call
                messages.append({
                    "role": "assistant",
                    "tool_calls": [{
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"}
                    }]
                })

                if norm_err:
                    # отдаём модели понятную ошибку вместо падения
                    messages.append({
                        "role": "tool", "tool_call_id": tc.id,
                        "content": json.dumps({
                            "isError": True,
                            "why": "argument_validation",
                            "message": norm_err,
                            "given_args": raw_args,
                            "normalized_args": args
                        }, ensure_ascii=False)
                    })
                    continue

                payload = await mcp_call(self.mcp_url, tc.function.name, args, timeout=self.tool_timeout)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(payload, ensure_ascii=False)
                })

        return last_text_reply or "(loop limit reached)"

# =========================
# CLI
# =========================

def main() -> None:
    try:
        settings = Settings.load()
    except Exception as e:
        print(f"⚠️ Config error: {e}", file=sys.stderr)
        sys.exit(2)

    client = OpenAI(api_key=settings.openrouter_api_key, base_url="https://openrouter.ai/api/v1")
    agent = MCPAgent(
        client=client,
        model=settings.model,
        mcp_url=settings.mcp_url,
        temperature=settings.temperature,
        max_steps=settings.max_steps,
        tool_timeout=settings.tool_timeout,
    )

    parser = argparse.ArgumentParser(description="OpenRouter ↔ Bright Data MCP agent (single-file)")
    parser.add_argument("prompt", type=str, help="User prompt")
    args = parser.parse_args()

    try:
        answer = asyncio.run(agent.run(args.prompt))
    except RuntimeError as e:
        # На случай запуска в окружении с уже работающим loop (редко в обычном скрипте)
        if "a running event loop" in str(e):
            # fallback
            loop = asyncio.get_event_loop()
            answer = loop.run_until_complete(agent.run(args.prompt))
        else:
            raise

    print("\n=== Answer ===\n")
    print(answer)

if __name__ == "__main__":
    main()

#python main.py "Найди официальный Q2 2025 пресс-релиз Tesla и дай 2–3 ключевых факта со ссылками."
#python main.py "вышел ли gpt-5 на момент 2025 года?"
#python main.py "какие есть полседние новости в игровой индустрии?"
#python main.py "какая сегодня дата?"
#python main.py "какие игры вышли и выйдут в этом году?"