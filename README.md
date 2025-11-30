# MCP BI Demo (Postgres + SQLAlchemy + FastMCP + OpenAI)

A minimal demo showing how to expose a Postgres-backed construction BI dataset through an MCP server built with the official Python SDK and SQLAlchemy. The stack runs via Docker Compose and is ready to plug into OpenAI's Responses API or Apps/Agents SDK with the `gpt-4.1-mini` (or `gpt-4o-mini`) model.

## Project layout

```
docker-compose.yml        # Postgres + MCP server services
.env.example              # sample environment (copy to .env)
server/
  Dockerfile              # Python MCP server container
  db_models.py            # SQLAlchemy models
  db_utils.py             # schema/seed helpers and readonly role
  mcp_db_server.py        # FastMCP server exposing DB tools
app/
  Dockerfile              # Lightweight chat gateway container
  main.py                 # Minimal FastAPI app that calls OpenAI + MCP
  pyproject.toml          # Gateway dependencies (FastAPI/OpenAI/Ollama)
```

## Architecture

```
client (curl, app, agent)
    |
    v
[Chat Gateway  : FastAPI + OpenAI SDK]
    |  (tool calls)
    v
OpenAI model (e.g., gpt-4.1-mini)
    |  (MCP over SSE @ http://mcp_server:8765/sse)
    v
[MCP DB Server : FastMCP + SQLAlchemy]  <-- long-lived container
    |
    v
Postgres
```

## Prerequisites

- Docker and Docker Compose
- Python 3.11+ (only needed if you want to run the server locally without Docker)

## Quickstart (Docker Compose)

1) Copy the environment template and fill in secrets:
```bash
cp .env.example .env
# edit .env to set POSTGRES_* credentials and your OpenAI API key
```
The `DB_URL_*` values in `.env.example` point to `db`, which is the hostname inside the Compose network. No changes needed if you run both services via `docker compose`.

2) Build and start everything:
```bash
 docker compose up --build
```
The `mcp_server` container will connect to `db`, create/seed tables, ensure the `readonly` role, and expose MCP tools over SSE at `http://mcp_server:8765/sse` (container entrypoint runs `python mcp_db_server.py` with `MCP_TRANSPORT=sse`). The `chat_gateway` container points to that URL via `MCP_SSE_URL`.

3) Check logs / quick sanity checks:
```bash
docker compose logs -f mcp_server
docker compose exec db psql -U admin -d construction -c "SELECT COUNT(*) FROM projects;"
docker compose exec db psql -U admin -d construction -c "SELECT COUNT(*) FROM cost_items;"
```

4) Wire your MCP client/agent to the running container:
- Preferred: use the SSE endpoint `http://localhost:8765/sse` (or `http://mcp_server:8765/sse` inside the Compose network). The server tells the client where to POST messages.
- If you need stdio instead, set `MCP_TRANSPORT=stdio` and point your client to a process (e.g., `docker compose exec mcp_server python mcp_db_server.py`).

5) Stop when done:
```bash
docker compose down
```

## Using the chat gateway (LLM + MCP)

- Runs as `chat_gateway` alongside `db` and `mcp_server` when you `docker compose up --build`.
- Choose provider via `.env`:
  - `LLM_PROVIDER=openai` (default): set `OPENAI_API_KEY`; `OPENAI_MODEL` defaults to `gpt-4.1-mini`.
  - `LLM_PROVIDER=ollama`: set `OLLAMA_BASE_URL` (default `http://host.docker.internal:11434/v1`, adjust if different) and `OLLAMA_MODEL` (default `qwen3-coder:30b`). `OLLAMA_API_KEY` is optional if your endpoint enforces auth.
- Endpoint: `POST http://localhost:8000/chat` with body:
  ```json
  { "question": "Which projects are over budget?" }
  ```
  Optional: `"model": "<model-name>"` to override the default for the chosen provider.
- MCP transport: the gateway requires `MCP_SSE_URL` (e.g., `http://mcp_server:8765/sse`) and only uses the long-lived MCP container; there is no local stdio fallback.

## Available MCP tools

- `get_schema()`: Returns table/column metadata and relationships.
- `run_sql(query: str, max_rows: int = 1000)`: Executes read-only SQL (SELECT-only, forbidden verbs blocked, row-limited).
- `project_cost_overview(project_name: str)`: Budget vs. actual vs. overrun for a project.
- `top_overruns(limit: int = 5, min_overrun_pct: float = 0.05)`: Projects exceeding budget by threshold.

## Wiring an OpenAI agent (high level)

- **OpenAI Apps / Agents SDK (recommended):** Declare your MCP server as a tool, set the model to `gpt-4.1-mini`, and run the MCP server as a child process (stdio) or HTTP bridge per the MCP SDK docs.
- **Custom script with Responses API:** Use the MCP Python SDK client to invoke server tools when the model issues tool calls.

Suggested system prompt snippet for your agent:

> You are a BI analyst for a construction company. Use the MCP DB tools to answer numeric questions. Never attempt to modify the database.

### Running the MCP server directly on your host (without the container)

Only do this if you want to debug the Python code locally. Two things to watch:
- Use `localhost` in your DB URLs (the `db` hostname only works from inside the Compose network).
- Make sure Postgres is already running (e.g., `docker compose up db`).

Steps:
1. Create/activate a Python 3.11+ venv.
2. Install deps: `pip install -e server`
3. Export DB URLs that point to your reachable Postgres (if you’re using the compose `db` service from the host, use `localhost`):
   ```bash
   export DB_URL_ADMIN=postgresql+psycopg2://admin:adminpass@localhost:5432/construction
   export DB_URL_RO=postgresql+psycopg2://readonly:readonlypass@localhost:5432/construction
   ```
4. Start the server from `server/`:
   ```bash
   cd server
   python -m mcp_db_server  # defaults to stdio; set MCP_TRANSPORT=sse to serve on http://localhost:8765/sse
   ```

On first run it will create/seed tables and ensure the readonly role.

### Example tool calls (handy for manual testing)

- Fetch schema:

  ```bash
  python - <<'PY'
  from db_utils import init_schema_and_seed, get_ro_engine
  from mcp_db_server import get_schema

  init_schema_and_seed()
  print(get_schema())
  PY
  ```

- Run a quick BI query for burn rate (requires Postgres running):

  ```bash
  python - <<'PY'
  from mcp_db_server import run_sql

  sql = """
  SELECT project_id, DATE_TRUNC('month', cost_date) AS month, SUM(amount_eur) AS total
  FROM cost_items
  GROUP BY project_id, month
  ORDER BY project_id, month
  """
  print(run_sql(sql, max_rows=20))
  PY
  ```

## Safety highlights

- All MCP tools use the `readonly` Postgres user.
- `run_sql` enforces SELECT-only queries and blocks common mutating verbs.
- Query results are capped by `max_rows` to avoid large responses.
- Secrets (DB credentials, OpenAI key) are supplied via `.env` and not committed.
