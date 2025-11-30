from __future__ import annotations

import re
import os
from typing import Any

from mcp.server.fastmcp import FastMCP
from sqlalchemy import inspect, text

from db_utils import get_ro_engine, init_schema_and_seed

init_schema_and_seed()
engine = get_ro_engine()

MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "stdio").lower()
MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.getenv("MCP_PORT", "8765" if MCP_TRANSPORT != "stdio" else "0"))
MCP_MOUNT_PATH = os.getenv("MCP_MOUNT_PATH", "/")
MCP_SSE_PATH = os.getenv("MCP_SSE_PATH", "/sse")
MCP_MESSAGE_PATH = os.getenv("MCP_MESSAGE_PATH", "/messages/")

mcp = FastMCP(
    "construction_bi_db",
    host=MCP_HOST,
    port=MCP_PORT,
    mount_path=MCP_MOUNT_PATH,
    sse_path=MCP_SSE_PATH,
    message_path=MCP_MESSAGE_PATH,
)

FORBIDDEN_SQL = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)


def ensure_safe_sql(query: str) -> None:
    if FORBIDDEN_SQL.search(query):
        raise ValueError("Only read-only SELECT queries are allowed.")


@mcp.tool()
def get_schema() -> dict[str, Any]:
    """Return database schema including columns and relationships."""

    inspector = inspect(engine)
    schema: dict[str, Any] = {"tables": {}, "relationships": []}

    for table_name in inspector.get_table_names():
        columns = inspector.get_columns(table_name)
        schema["tables"][table_name] = {
            "columns": {column["name"]: str(column["type"]) for column in columns}
        }

        foreign_keys = inspector.get_foreign_keys(table_name)
        for fk in foreign_keys:
            for local_col, remote_col in zip(
                fk["constrained_columns"], fk["referred_columns"]
            ):
                schema["relationships"].append(
                    {
                        "from_table": table_name,
                        "from_column": local_col,
                        "to_table": fk["referred_table"],
                        "to_column": remote_col,
                    }
                )

    schema["dialect"] = "postgresql"
    return schema


@mcp.tool()
def run_sql(query: str, max_rows: int = 1000) -> dict[str, Any]:
    """Execute a read-only SQL query and return results."""

    ensure_safe_sql(query)
    query_stripped = query.strip().rstrip(";")
    if not query_stripped.lower().startswith("select"):
        raise ValueError("Only SELECT queries are allowed.")

    rows: list[dict[str, Any]] = []
    with engine.connect() as conn:
        result = conn.execute(text(query_stripped))
        keys = result.keys()
        for index, row in enumerate(result):
            if index >= max_rows:
                break
            rows.append(dict(zip(keys, row)))

    return {"rows": rows, "row_count": len(rows)}


@mcp.tool()
def project_cost_overview(project_name: str) -> dict[str, Any]:
    """Return budget, total cost, and overrun details for a project."""

    sql = """
    SELECT
      p.name,
      p.budget_eur,
      COALESCE(SUM(c.amount_eur), 0) AS actual_cost_eur,
      COALESCE(SUM(c.amount_eur), 0) - p.budget_eur AS overrun_eur
    FROM projects p
    LEFT JOIN cost_items c ON c.project_id = p.id
    WHERE p.name = :name
    GROUP BY p.id
    """

    with engine.connect() as conn:
        result = conn.execute(text(sql), {"name": project_name}).mappings().first()
        if not result:
            return {"error": f"Project '{project_name}' not found."}
        return dict(result)


@mcp.tool()
def top_overruns(limit: int = 5, min_overrun_pct: float = 0.05) -> dict[str, Any]:
    """Return projects whose actual costs exceed budget by a threshold."""

    sql = """
    SELECT
      p.name,
      p.budget_eur,
      COALESCE(SUM(c.amount_eur), 0) AS actual_cost_eur,
      CASE
        WHEN p.budget_eur = 0 THEN NULL
        ELSE (COALESCE(SUM(c.amount_eur), 0) - p.budget_eur) / p.budget_eur
      END AS overrun_pct,
      COALESCE(SUM(c.amount_eur), 0) - p.budget_eur AS overrun_eur
    FROM projects p
    LEFT JOIN cost_items c ON c.project_id = p.id
    GROUP BY p.id
    HAVING (COALESCE(SUM(c.amount_eur), 0) - p.budget_eur) / NULLIF(p.budget_eur, 0) >= :min_overrun_pct
    ORDER BY overrun_pct DESC NULLS LAST
    LIMIT :limit
    """

    with engine.connect() as conn:
        results = conn.execute(
            text(sql), {"limit": limit, "min_overrun_pct": min_overrun_pct}
        ).mappings()
        rows = [dict(row) for row in results]
    return {"rows": rows, "row_count": len(rows)}
def run_server() -> None:
    # Default to stdio for local demos; switch to SSE when running as a service.
    mcp.run(transport=MCP_TRANSPORT)


if __name__ == "__main__":
    run_server()
