from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import warnings
import os
import openai
import json
from pathlib import Path

load_dotenv()

@tool
def calculator(a:float, b:float)->str:
    """this is useful for doing simple calculations"""
    print("tool has been called")
    return f"the result of {a} + {b} is {a+b}"

@tool
def todo_manager(action: str, task: str = "", task_id: int | None = None) -> str:
    """Manage a simple local todo list.

    Actions:
    - add: provide `task` to add a todo
    - list: returns all todos
    - complete: provide `task_id` to mark a todo done
    """
    path = Path("agent_todos.json")
    try:
        if path.exists():
            todos = json.loads(path.read_text())
        else:
            todos = []

        action = (action or "").strip().lower()
        if action == "add":
            if not task:
                return "Usage: action='add', provide task string"
            new_id = max((t.get("id", 0) for t in todos), default=0) + 1
            todos.append({"id": new_id, "task": task, "done": False})
            path.write_text(json.dumps(todos, indent=2))
            return f"Added todo #{new_id}: {task}"

        if action == "list":
            if not todos:
                return "No todos."
            lines = []
            for t in todos:
                mark = "x" if t.get("done") else " "
                lines.append(f"{t['id']}. [{mark}] {t['task']}")
            return "\n".join(lines)

        if action == "complete":
            if task_id is None:
                return "Usage: action='complete', provide task_id"
            for t in todos:
                if t.get("id") == task_id:
                    t["done"] = True
                    path.write_text(json.dumps(todos, indent=2))
                    return f"Marked todo #{task_id} complete."
            return f"Todo #{task_id} not found."

        return "Unknown action. Valid: add, list, complete"
    except Exception as e:
        return f"todo_manager error: {e}"

@tool
def wiki_lookup(query: str, sentences: int = 2) -> str:
    """Return a short Wikipedia summary for `query`."""
    import requests
    from requests.utils import requote_uri
    try:
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + requote_uri(query)
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            extract = data.get("extract", "")
            if not extract:
                return "No summary found on Wikipedia."
            # return first N sentences
            parts = extract.split(". ")
            return ". ".join(parts[:sentences]).strip() + ("" if len(parts) <= sentences else "...")
        elif r.status_code == 404:
            return "No Wikipedia page found for that query."
        else:
            return f"Wikipedia lookup failed: HTTP {r.status_code}"
    except Exception as ex:
        return f"Wiki lookup error: {ex}"
    
@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """Convert `value` from `from_unit` to `to_unit`.

    Supports lengths (m, cm, mm, km, in, ft, yd, mi),
    weights (kg, g, mg, lb, oz), and temperatures (C, F, K).
    """
    try:
        fu = from_unit.strip().lower()
        tu = to_unit.strip().lower()

        # length (base: meter)
        lengths = {
            "m": 1.0, "cm": 0.01, "mm": 0.001, "km": 1000.0,
            "in": 0.0254, "ft": 0.3048, "yd": 0.9144, "mi": 1609.344,
        }

        # weight (base: kilogram)
        weights = {
            "kg": 1.0, "g": 0.001, "mg": 1e-6, "lb": 0.45359237, "oz": 0.028349523125,
        }

        # temperature handled separately
        if fu in ("c", "f", "k") and tu in ("c", "f", "k"):
            v = float(value)
            # convert from -> c
            if fu == "c":
                c = v
            elif fu == "f":
                c = (v - 32) * 5.0 / 9.0
            else:  # k
                c = v - 273.15

            # c -> target
            if tu == "c":
                out = c
            elif tu == "f":
                out = c * 9.0 / 5.0 + 32
            else:  # k
                out = c + 273.15

            return f"{value} {from_unit} = {out} {to_unit}"

        if fu in lengths and tu in lengths:
            meters = float(value) * lengths[fu]
            out = meters / lengths[tu]
            return f"{value} {from_unit} = {out} {to_unit}"

        if fu in weights and tu in weights:
            kg = float(value) * weights[fu]
            out = kg / weights[tu]
            return f"{value} {from_unit} = {out} {to_unit}"

        return f"Unsupported unit conversion: {from_unit} -> {to_unit}"
    except Exception as e:
        return f"unit_converter error: {e}"
    

warnings.filterwarnings(
    "ignore",
    message=r".*create_react_agent has been moved.*",
)

# Verify OPENAI_API_KEY is available (masked when shown)
_openai_key = os.getenv("OPENAI_API_KEY")
if _openai_key:
    masked = _openai_key[:4] + "..." + _openai_key[-4:]
    print(f"OPENAI_API_KEY found: {masked}")
else:
    print("Warning: OPENAI_API_KEY not found in environment.\nPlease ensure .env is present and contains OPENAI_API_KEY.")

def main():
    model = ChatOpenAI(temperature=0)

    tools=[calculator, todo_manager, unit_converter, wiki_lookup]
    agent_executor = create_react_agent(model=model, tools=tools)

    print("Welcome I'm your assistant for today. How can i help you!")
    print("Type 'exit' to quit the program.")

    while True:
        user_input = input("\nYou:").strip()

        if user_input.lower() == 'exit':
            print("if you need any help in the future, just ask. Goodbye!")
            break

        print("\nassistant:", end="")
        try:
            for chunk in agent_executor.stream(
                {"messages": [HumanMessage(content=user_input)]}
            ):
                if "agent" in chunk and "messages" in chunk["agent"]:
                    for message in chunk["agent"]["messages"]:
                        print(message.content, end="")
        except openai.RateLimitError as e:
            # show detailed error returned by OpenAI (without revealing API key)
            print(f"\n[OpenAI RateLimitError] {e}", end="")
            if hasattr(e, 'response') and getattr(e.response, 'text', None):
                try:
                    print(f"\nDetails: {e.response.text}", end="")
                except Exception:
                    pass
            print("\nCheck your OpenAI account billing and quota: https://platform.openai.com/account/billing/overview", end="")
        except Exception as e:
            print(f"\n[Error] Unexpected error: {e}", end="")

        print()

if __name__ == "__main__":
    main()
    

        