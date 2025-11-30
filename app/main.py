from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel

from mcp import StdioServerParameters, stdio_client
from mcp.client.session import ClientSession


SYSTEM_PROMPT = """You are a BI assistant for a construction company.

Goals:
- Answer using the Postgres database via MCP tools only; no guessing.
- Think stepwise: restate intent briefly, identify relevant tables/columns, then query.
- Use only read-only SQL.
- Handle fuzzy user input by generating a few alternative spellings/keywords (singular/plural, typos, accents, substrings). Use ILIKE with wildcards or unaccent/LOWER if available to discover the best match before the final query.
- Prefer tight filters, aggregates, and ordering; add LIMITs to exploratory lookups.
- Return concise numeric answers with minimal explanation; include units/time frames; mention if data is partial.
- If nothing is found, state that and share the closest matches you probed.

Behaviors:
- Rely on the provided schema and listed tools; do not invent tables/columns.
- Keep tool calls minimal and deterministic; avoid expensive full scans unless necessary.
- Never modify data; queries must be read-only.

After formulating your answer reply in the same language of the question (most likely Italian).
"""

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3-coder:30b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434/v1")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")  # ollama ignores key by default

app = FastAPI(title="MCP BI Chat Gateway", version="0.1.0")


class ChatRequest(BaseModel):
    question: str
    model: str | None = None


def _get_llm_client_and_model(model_override: str | None) -> tuple[AsyncOpenAI, str]:
    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")
        return AsyncOpenAI(api_key=OPENAI_API_KEY), (model_override or OPENAI_MODEL)

    if LLM_PROVIDER == "ollama":
        client = AsyncOpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)
        return client, (model_override or OLLAMA_MODEL)

    raise RuntimeError(f"Unsupported LLM_PROVIDER '{LLM_PROVIDER}'. Use 'openai' or 'ollama'.")


def _tool_to_openai(tool: Any) -> dict[str, Any]:
    """Convert an MCP tool definition to an OpenAI tool schema."""
    params = tool.inputSchema or {"type": "object", "properties": {}, "additionalProperties": False}
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": params,
        },
    }


def _tool_result_to_text(result: Any) -> str:
    if getattr(result, "isError", False):
        return json.dumps({"error": result.error})
    if getattr(result, "structuredContent", None) is not None:
        return json.dumps(result.structuredContent)

    chunks: list[str] = []
    for item in getattr(result, "content", []) or []:
        if getattr(item, "type", "") == "text":
            chunks.append(item.text)
        else:
            chunks.append(item.model_dump_json())
    return "\n".join(chunks) if chunks else "{}"


async def _chat_with_tools(question: str, model: str) -> str:
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "mcp_db_server"],
        # Pass through DB env so the child MCP server can connect.
        env={k: v for k, v in os.environ.items() if k.startswith("DB_") or k.startswith("POSTGRES_")},
        cwd="/app",
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            tool_defs = [_tool_to_openai(tool) for tool in tools]

            # Fetch schema once and pass it to the model to reduce hallucinated columns/tables.
            schema_note = ""
            try:
                schema_result = await session.call_tool("get_schema", arguments={})
                schema_note = json.dumps(schema_result.structuredContent or {}, indent=2)
            except Exception:
                schema_note = "Schema unavailable (tool call failed). Rely only on listed MCP tools."

            messages: list[dict[str, Any]] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "system", "content": f"Database schema:\n{schema_note}\nUse only these tables/columns."},
                {"role": "user", "content": question},
            ]

            llm_client, resolved_model = _get_llm_client_and_model(model)

            # Tool-call loop with a small hop limit for safety.
            for _ in range(6):
                completion = await llm_client.chat.completions.create(
                    model=resolved_model,
                    messages=messages,
                    tools=tool_defs,
                )
                message = completion.choices[0].message

                if message.tool_calls:
                    # Record the assistant's tool calls for the transcript.
                    messages.append(
                        {
                            "role": "assistant",
                            "content": message.content,
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": tc.type,
                                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                                }
                                for tc in message.tool_calls
                            ],
                        }
                    )

                    for tc in message.tool_calls:
                        try:
                            args = json.loads(tc.function.arguments or "{}")
                        except json.JSONDecodeError as exc:  # pragma: no cover - model bug
                            args = {}
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tc.id,
                                    "content": f"Invalid tool arguments JSON: {exc}",
                                }
                            )
                            continue

                        try:
                            result = await session.call_tool(tc.function.name, arguments=args)
                            result_text = _tool_result_to_text(result)
                        except Exception as exc:
                            result_text = f"Tool '{tc.function.name}' failed: {exc}"

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": result_text,
                            }
                        )
                    continue

                return message.content or ""

            raise RuntimeError("Tool loop exceeded without reaching a final answer.")


@app.post("/chat")
async def chat(req: ChatRequest) -> dict[str, str]:
    try:
        answer = await _chat_with_tools(req.question, req.model or "")
        return {"model": req.model or (OLLAMA_MODEL if LLM_PROVIDER == "ollama" else OPENAI_MODEL), "answer": answer}
    except Exception as exc:  # pragma: no cover - bubbled to API
        raise HTTPException(status_code=500, detail=str(exc)) from exc
