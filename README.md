# MCP BI Demo (Postgres + SQLAlchemy + FastMCP + OpenAI)

A minimal demo showing how to expose a Postgres-backed construction BI dataset through an MCP server built with the official Python SDK and SQLAlchemy. The stack runs via Docker Compose and is ready to plug into OpenAI's Responses API or Apps/Agents SDK with the `gpt-4.1-mini` (or `gpt-4o-mini`) model.

## Project layout

```
docker-compose.yml        # Postgres + MCP server services
.env.example              # sample environment (copy to .env)
server/
  Dockerfile              # Python MCP server container
  requirements.txt        # MCP + SQLAlchemy dependencies
  db_models.py            # SQLAlchemy models
  db_utils.py             # schema/seed helpers and readonly role
  mcp_db_server.py        # FastMCP server exposing DB tools
```

## Prerequisites

- Docker and Docker Compose
- Python 3.11+ (only needed if you want to run the server locally without Docker)

## Setup

1. Copy the environment template and fill in secrets:

   ```bash
   cp .env.example .env
   # edit .env to set POSTGRES_* credentials and your OpenAI API key
   ```

2. Build and start the stack:

   ```bash
   docker compose up --build
   ```

   On first start the MCP server will:
   - create tables and seed ~5 projects with costs and deliveries,
   - create a `readonly` Postgres role with SELECT-only privileges,
   - expose MCP tools over stdio for clients.

3. Inspect the startup logs:

   ```bash
   docker compose logs -f mcp_server
   ```

4. (Optional) Verify the seeded data quickly:

   ```bash
   docker compose exec db psql -U admin -d construction -c "SELECT COUNT(*) FROM projects;"
   docker compose exec db psql -U admin -d construction -c "SELECT COUNT(*) FROM cost_items;"
   ```

5. Stop the stack when you are done:

   ```bash
   docker compose down
   ```

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

### Local MCP server usage without Docker

If you want to run the MCP server directly on your machine (for development):

1. Create and activate a Python 3.11+ virtual environment.
2. Install dependencies:

   ```bash
   pip install -r server/requirements.txt
   ```

3. Ensure Postgres is running and `DB_URL_ADMIN` / `DB_URL_RO` are set in your shell (use the same values as `.env`).
4. Start the server from the `server/` directory:

   ```bash
   cd server
   python -m mcp_db_server
   ```

The server runs over stdio, so you can attach any MCP-compatible client. The `init_schema_and_seed()` call will create tables, seed data, and ensure the readonly role on first run.

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
