import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from openai import AzureOpenAI
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
TASKS_FILE = DATA_DIR / "tasks.json"
CONVERSATIONS_FILE = DATA_DIR / "conversations.json"

AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY", "")
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# ---------------------------------------------------------------------------
# Persistent JSON helpers
# ---------------------------------------------------------------------------

DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text())
    return default


def _save_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2, default=str))


def load_tasks() -> list[dict]:
    return _load_json(TASKS_FILE, [])


def save_tasks(tasks: list[dict]):
    _save_json(TASKS_FILE, tasks)


def load_conversations() -> list[dict]:
    return _load_json(CONVERSATIONS_FILE, [])


def save_conversations(convos: list[dict]):
    _save_json(CONVERSATIONS_FILE, convos)


# ---------------------------------------------------------------------------
# Azure OpenAI client
# ---------------------------------------------------------------------------


def get_openai_client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )


def build_system_prompt() -> str:
        return f"""\
You are an intelligent task management assistant. The current date/time is {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}.

Your capabilities:
1. Help users create, update, delete, and organize tasks.
2. Break complex tasks into smaller actionable subtasks.
3. Prioritize tasks based on urgency, importance, and deadlines.
4. Identify dependencies between tasks.
5. Suggest related tasks the user may have overlooked.

When the user asks you to create or modify tasks, respond with a JSON block inside
```json ... ``` fences that contains an "actions" array. Each action object has:
- "action": one of "create", "update", "delete"
- "task": an object with relevant fields (title, description, priority, status, deadline, parent_id, depends_on, tags).
    For update/delete include the "id" field.

Priority values: "high", "medium", "low"
Status values: "todo", "in-progress", "done"

Always include a friendly human-readable message OUTSIDE the JSON block as well.
If the user is just chatting or asking questions, respond normally without JSON.
"""

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str


class TaskCreate(BaseModel):
    title: str
    description: str = ""
    priority: str = "medium"
    status: str = "todo"
    deadline: str | None = None
    parent_id: str | None = None
    depends_on: list[str] = []
    tags: list[str] = []


class TaskUpdate(BaseModel):
    title: str | None = None
    description: str | None = None
    priority: str | None = None
    status: str | None = None
    deadline: str | None = None
    parent_id: str | None = None
    depends_on: list[str] | None = None
    tags: list[str] | None = None


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="AI Task Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Routes – UI
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = STATIC_DIR / "index.html"
    return HTMLResponse(index_path.read_text())


# ---------------------------------------------------------------------------
# Routes – Tasks CRUD
# ---------------------------------------------------------------------------


@app.get("/api/tasks")
async def list_tasks():
    return load_tasks()


@app.post("/api/tasks", status_code=201)
async def create_task(body: TaskCreate):
    tasks = load_tasks()
    task = {
        "id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **body.model_dump(),
    }
    tasks.append(task)
    save_tasks(tasks)
    return task


@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    for t in load_tasks():
        if t["id"] == task_id:
            return t
    raise HTTPException(404, "Task not found")


@app.patch("/api/tasks/{task_id}")
async def update_task(task_id: str, body: TaskUpdate):
    tasks = load_tasks()
    for t in tasks:
        if t["id"] == task_id:
            updates = body.model_dump(exclude_none=True)
            t.update(updates)
            t["updated_at"] = datetime.now(timezone.utc).isoformat()
            save_tasks(tasks)
            return t
    raise HTTPException(404, "Task not found")


@app.delete("/api/tasks/{task_id}")
async def delete_task(task_id: str):
    tasks = load_tasks()
    new_tasks = [t for t in tasks if t["id"] != task_id]
    if len(new_tasks) == len(tasks):
        raise HTTPException(404, "Task not found")
    save_tasks(new_tasks)
    return {"deleted": task_id}


# ---------------------------------------------------------------------------
# Routes – Chat / AI
# ---------------------------------------------------------------------------


def _apply_ai_actions(actions: list[dict]) -> list[dict]:
    """Apply create/update/delete actions returned by the AI and return affected tasks."""
    tasks = load_tasks()
    affected: list[dict] = []

    for act in actions:
        action_type = act.get("action")
        task_data = act.get("task", {})

        if action_type == "create":
            new_task = {
                "id": str(uuid.uuid4()),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "title": task_data.get("title", "Untitled"),
                "description": task_data.get("description", ""),
                "priority": task_data.get("priority", "medium"),
                "status": task_data.get("status", "todo"),
                "deadline": task_data.get("deadline"),
                "parent_id": task_data.get("parent_id"),
                "depends_on": task_data.get("depends_on", []),
                "tags": task_data.get("tags", []),
            }
            tasks.append(new_task)
            affected.append(new_task)

        elif action_type == "update":
            tid = task_data.get("id")
            for t in tasks:
                if t["id"] == tid:
                    for k, v in task_data.items():
                        if k != "id":
                            t[k] = v
                    t["updated_at"] = datetime.now(timezone.utc).isoformat()
                    affected.append(t)
                    break

        elif action_type == "delete":
            tid = task_data.get("id")
            before = len(tasks)
            tasks = [t for t in tasks if t["id"] != tid]
            if len(tasks) < before:
                affected.append({"id": tid, "deleted": True})

    save_tasks(tasks)
    return affected


def _extract_json_actions(text: str) -> list[dict] | None:
    """Try to extract a JSON actions block from the AI response."""
    import re

    m = re.search(r"```json\s*(\{.*?})\s*```", text, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
        return data.get("actions")
    except (json.JSONDecodeError, AttributeError):
        return None


@app.post("/api/chat")
async def chat(body: ChatRequest):
    conversations = load_conversations()
    tasks = load_tasks()

    # Build message context
    messages: list[dict] = [{"role": "system", "content": build_system_prompt()}]

    # Include current tasks summary
    if tasks:
        task_summary = json.dumps(
            [{"id": t["id"], "title": t["title"], "status": t["status"],
              "priority": t["priority"], "deadline": t.get("deadline")}
             for t in tasks],
            indent=2,
        )
        messages.append({
            "role": "system",
            "content": f"Current tasks:\n{task_summary}",
        })

    # Keep last 20 conversation messages for context
    for msg in conversations[-20:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": body.message})

    # Call Azure OpenAI
    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            temperature=0.7,
            max_tokens=2048,
        )
        reply = response.choices[0].message.content or ""
    except Exception as e:
        raise HTTPException(502, f"AI service error: {e}")

    # Persist conversation
    conversations.append({"role": "user", "content": body.message, "ts": datetime.now(timezone.utc).isoformat()})
    conversations.append({"role": "assistant", "content": reply, "ts": datetime.now(timezone.utc).isoformat()})
    # Keep conversation history bounded
    if len(conversations) > 100:
        conversations = conversations[-100:]
    save_conversations(conversations)

    # Process any task actions from the AI
    affected_tasks: list[dict] = []
    actions = _extract_json_actions(reply)
    if actions:
        affected_tasks = _apply_ai_actions(actions)

    return {
        "reply": reply,
        "affected_tasks": affected_tasks,
        "tasks": load_tasks(),
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
