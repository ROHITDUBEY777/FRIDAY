import json
import os
import time
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["GROQ_API_KEY"]
API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"  # confirm at console.groq.com/docs/models

SYSTEM_PROMPT = (
    "You are JARVIS, a concise personal voice assistant. "
    "Reply in short plain spoken sentences, no markdown or lists."
)


def call_llm(messages, tools, retries=3):
    for attempt in range(retries):
        r = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                "max_tokens": 1024,
            },
            timeout=60,
        )

        # Free tier rate limit: wait and retry
        if r.status_code == 429:
            wait = float(r.headers.get("retry-after", 2 ** attempt))
            print(f"Rate limited, waiting {wait}s...")
            time.sleep(wait)
            continue

        if not r.ok:
            print("STATUS:", r.status_code)
            print("BODY:", r.text)
            r.raise_for_status()

        return r.json()["choices"][0]["message"]

    raise RuntimeError("Still rate limited after retries.")


def run_agent(user_msg, history, tools, tool_impls, max_steps=8):
    if not history or history[0].get("role") != "system":
        history.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    history.append({"role": "user", "content": user_msg})

    for _ in range(max_steps):
        msg = call_llm(history, tools)
        tool_calls = msg.get("tool_calls") or []

        assistant_turn = {"role": "assistant", "content": msg.get("content") or ""}
        if tool_calls:
            assistant_turn["tool_calls"] = tool_calls
        history.append(assistant_turn)

        # No tool calls means this is the final answer
        if not tool_calls:
            return msg.get("content") or ""

        for tc in tool_calls:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
                out = tool_impls[name](**args)
            except Exception as e:
                out = f"Tool error: {e}"
            history.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": str(out),
            })

    return "I couldn't finish that in time."


if __name__ == "__main__":
    tools = [{
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current local time.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }]
    tool_impls = {"get_time": lambda: datetime.now().strftime("%I:%M %p")}

    history = []
    print(run_agent("What time is it?", history, tools, tool_impls))