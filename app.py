from flask import Flask, request, render_template, jsonify
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFacePipeline
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import warnings
import os
import json
from pathlib import Path
from transformers import pipeline
from types import SimpleNamespace
import threading
import time
from openai import OpenAI

load_dotenv()

# Suppress warnings
warnings.filterwarnings("ignore", message=r".*create_react_agent has been moved.*")

# Tool definitions (copied from main.py)
@tool
def calculator(a: float, b: float) -> str:
    """this is useful for doing simple calculations"""
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

@tool
def palindrome_checker(text: str) -> str:
    """Check if the given text is a palindrome (ignores case and spaces)."""
    try:
        cleaned = ''.join(c.lower() for c in text if c.isalnum())
        is_pal = cleaned == cleaned[::-1]
        return f"'{text}' is {'a palindrome' if is_pal else 'not a palindrome'}."
    except Exception as e:
        return f"palindrome_checker error: {e}"

@tool
def random_number_generator(min_val: int = 1, max_val: int = 100) -> str:
    """Generate a random integer between min_val and max_val (inclusive)."""
    try:
        import random
        num = random.randint(min_val, max_val)
        return f"Random number between {min_val} and {max_val}: {num}"
    except Exception as e:
        return f"random_number_generator error: {e}"

@tool
def text_reverser(text: str) -> str:
    """Reverse the given text."""
    try:
        return f"Reversed: '{text[::-1]}'"
    except Exception as e:
        return f"text_reverser error: {e}"

@tool
def regex_tool(pattern: str, text: str, action: str = "findall") -> str:
    """Perform regex operations on text. Actions: findall, search, sub."""
    try:
        import re
        if action == "findall":
            matches = re.findall(pattern, text)
            return f"Matches: {matches}"
        elif action == "search":
            match = re.search(pattern, text)
            return f"Match: {match.group() if match else 'No match'}"
        elif action == "sub":
            result = re.sub(pattern, "", text)
            return f"Replaced: {result}"
        else:
            return "Invalid action. Use findall, search, or sub."
    except Exception as e:
        return f"regex_tool error: {e}"

@tool
def json_validator(json_str: str) -> str:
    """Validate and pretty-print JSON string."""
    try:
        import json
        parsed = json.loads(json_str)
        pretty = json.dumps(parsed, indent=2)
        return f"Valid JSON:\n{pretty}"
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"
    except Exception as e:
        return f"json_validator error: {e}"

@tool
def base64_tool(text: str, action: str = "encode") -> str:
    """Encode or decode text using Base64. Actions: encode, decode."""
    try:
        import base64
        if action == "encode":
            encoded = base64.b64encode(text.encode()).decode()
            return f"Encoded: {encoded}"
        elif action == "decode":
            decoded = base64.b64decode(text).decode()
            return f"Decoded: {decoded}"
        else:
            return "Invalid action. Use encode or decode."
    except Exception as e:
        return f"base64_tool error: {e}"

# Initialize agent placeholder; load model in background to avoid blocking server start
agent_executor = None
_model_loading = False
model_info = {"source": None, "name": None}

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
        
def load_model_background():
    global agent_executor, _model_loading, model_info
    if agent_executor is not None or _model_loading:
        return
    _model_loading = True
    try:
        # this may take time (downloads model on first run)
        print("[model loader] Starting model download/load (this may take a minute)...")

        # If HF_TOKEN is present, prefer using the Hugging Face Inference Router
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            try:
                client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=hf_token)

                class HFAdapterRemote:
                    def __init__(self, client, model_name="openai/gpt-oss-120b"):
                        self.client = client
                        self.model_name = model_name

                    def stream(self, model_input):
                        try:
                            msgs = model_input.get("messages") if isinstance(model_input, dict) else None
                            user_text = getattr(msgs[0], "content", str(msgs[0])) if msgs and len(msgs) > 0 else ""

                            completion = self.client.chat.completions.create(
                                model=self.model_name,
                                messages=[{"role": "user", "content": user_text}],
                            )
                            text = completion.choices[0].message.content
                            message = SimpleNamespace(content=text)
                            yield {"agent": {"messages": [message]}}
                        except Exception as e:
                            errstr = str(e)
                            print(f"[remote adapter] error: {errstr}")
                            # detect permission / 403 error and fallback to local model
                            if "403" in errstr or "insufficient permissions" in errstr.lower():
                                model_info["error"] = errstr
                                # attempt to instantiate a lightweight local fallback
                                try:
                                    import torch
                                    device = 0 if torch.cuda.is_available() else -1
                                except Exception:
                                    device = -1

                                try:
                                    local_pipe = pipeline("text-generation", model="distilgpt2", device=device)

                                    class LocalAdapter:
                                        def __init__(self, pipe):
                                            self.pipe = pipe

                                        def stream(self, model_input_inner):
                                            try:
                                                msgs = model_input_inner.get("messages") if isinstance(model_input_inner, dict) else None
                                                user_text_inner = getattr(msgs[0], "content", str(msgs[0])) if msgs and len(msgs) > 0 else ""
                                                prompt = (
                                                    "You are a helpful assistant. Provide a concise, direct answer.\n"
                                                    f"User: {user_text_inner}\n"
                                                    "Assistant:"
                                                )
                                                out = self.pipe(
                                                    prompt,
                                                    max_new_tokens=120,
                                                    do_sample=False,
                                                    temperature=0.2,
                                                    top_p=0.95,
                                                    repetition_penalty=1.1,
                                                )
                                                gen = out[0].get("generated_text") if isinstance(out, list) and isinstance(out[0], dict) else str(out)
                                                if isinstance(gen, str) and gen.startswith(prompt):
                                                    gen = gen[len(prompt):]
                                                gen = gen.strip()
                                                message = SimpleNamespace(content=gen)
                                                yield {"agent": {"messages": [message]}}
                                            except Exception as e2:
                                                message = SimpleNamespace(content=f"[Fallback model error] {e2}")
                                                yield {"agent": {"messages": [message]}}

                                    local_adapter = LocalAdapter(local_pipe)
                                    agent_executor = local_adapter
                                    model_info["source"] = "local"
                                    model_info["name"] = "distilgpt2"
                                    print("[model loader] Remote 403 detected — falling back to local distilgpt2")
                                    for chunk in local_adapter.stream(model_input):
                                        yield chunk
                                    return
                                except Exception as e_local:
                                    print(f"[fallback error] {e_local}")
                                    message = SimpleNamespace(content=f"[Remote model error] {errstr}")
                                    yield {"agent": {"messages": [message]}}
                            else:
                                message = SimpleNamespace(content=f"[Remote model error] {e}")
                                yield {"agent": {"messages": [message]}}

                tools = [calculator, todo_manager, unit_converter, palindrome_checker, random_number_generator, text_reverser, regex_tool, json_validator, base64_tool]
                agent_executor = HFAdapterRemote(client)
                model_info["source"] = "remote"
                model_info["name"] = "openai/gpt-oss-120b"
                print("[model loader] Using Hugging Face Inference API (remote model: openai/gpt-oss-120b).")
                return
            except Exception as e:
                print(f"Remote model initialization failed: {e}")

        # Fallback: local pipeline (smaller model to reduce download size and memory usage)
        try:
            import torch
            device = 0 if torch.cuda.is_available() else -1
        except Exception:
            device = -1

        hf_pipeline = pipeline("text-generation", model="distilgpt2", device=device)

        # Adapter to present a minimal streaming interface expected by the app
        class HFAdapter:
            def __init__(self, pipe, tools=None):
                self.pipe = pipe
                self.tools = tools or []

            def stream(self, model_input):
                # model_input expected like {"messages":[HumanMessage(...)]}
                try:
                    msgs = model_input.get("messages") if isinstance(model_input, dict) else None
                    if msgs and len(msgs) > 0:
                        user_text = getattr(msgs[0], "content", str(msgs[0]))
                    else:
                        user_text = ""

                    prompt = (
                        "You are a helpful assistant. Provide a concise, direct answer.\n"
                        f"User: {user_text}\n"
                        "Assistant:"
                    )

                    out = self.pipe(
                        prompt,
                        max_new_tokens=120,
                        do_sample=False,
                        temperature=0.2,
                        top_p=0.95,
                        repetition_penalty=1.1,
                    )

                    gen = out[0].get("generated_text") if isinstance(out, list) and isinstance(out[0], dict) else str(out)
                    if isinstance(gen, str) and gen.startswith(prompt):
                        gen = gen[len(prompt):]
                    gen = gen.strip()

                    message = SimpleNamespace(content=gen)
                    chunk = {"agent": {"messages": [message]}}
                    yield chunk
                except Exception as e:
                    message = SimpleNamespace(content=f"[Model error] {e}")
                    yield {"agent": {"messages": [message]}}

        tools = [calculator, todo_manager, unit_converter, palindrome_checker, random_number_generator, text_reverser, regex_tool, json_validator, base64_tool]
        agent_executor = HFAdapter(hf_pipeline, tools=tools)
        model_info["source"] = "local"
        model_info["name"] = "distilgpt2"
        print("[model loader] Local model adapter ready (distilgpt2 fallback).")
    except Exception as e:
        print(f"Model load error: {e}")
    finally:
        _model_loading = False

# Start background loader thread
threading.Thread(target=load_model_background, daemon=True).start()

app = Flask(__name__)

# UI template moved to templates/index.html

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('message', '').strip()
    if not user_input:
        return jsonify({'response': 'Please enter a message.'})

    try:
        # If model still loading, inform the user
        if agent_executor is None:
            return jsonify({'response': 'Model is still loading. Please wait a moment and try again.'})
        response_text = ""
        for chunk in agent_executor.stream({"messages": [HumanMessage(content=user_input)]}):
            if "agent" in chunk and "messages" in chunk["agent"]:
                for message in chunk["agent"]["messages"]:
                    response_text += message.content
        return jsonify({'response': response_text})
    except Exception as e:
        return jsonify({'response': f'[Error] {e}'})


@app.route('/status', methods=['GET'])
def status():
    """Return whether the model is ready."""
    try:
        ready = agent_executor is not None
        return jsonify({
            'ready': bool(ready),
            'loading': not ready and _model_loading,
            'error': None if ready or _model_loading else 'model not initialized',
            'model': model_info.get('name'),
            'model_source': model_info.get('source'),
            'model_error': model_info.get('error')
        })
    except Exception as e:
        return jsonify({'ready': False, 'loading': False, 'error': str(e)})

if __name__ == '__main__':
    print("Starting server on http://0.0.0.0:8080")
    app.run(host='0.0.0.0', port=8080, debug=True)