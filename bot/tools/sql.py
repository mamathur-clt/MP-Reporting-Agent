"""
Generic read-only Databricks SQL tool for the agent.

Provides the OpenAI function-calling schema and executor.
"""

import json

from bot.db import execute_readonly_sql

TOOL_DEFINITION: dict = {
    "type": "function",
    "function": {
        "name": "run_databricks_sql",
        "description": (
            "Run a read-only SQL SELECT query against Databricks. "
            "Only SELECT / WITH queries are allowed. "
            "Key schemas: energy_prod.energy (v_sessions, v_carts, v_orders, v_calls, v_orders_gcv), "
            "energy_prod.data_science (mp_session_level_query), "
            "lakehouse_production.common (gsc_search_analytics_d_1), "
            "energy_prod.energy.rpt_texas_daily_pacing. "
            "Results are capped at 500 rows."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "SQL SELECT query to execute.",
                },
                "explanation": {
                    "type": "string",
                    "description": "One-line explanation of what this query does.",
                },
            },
            "required": ["query", "explanation"],
        },
    },
}


def execute(args_json: str) -> tuple[str, str]:
    """
    Execute a run_databricks_sql tool call.

    Returns (result_text, explanation).
    """
    args = json.loads(args_json)
    explanation = args.get("explanation", "")
    query = args.get("query", "")
    result_str, _ = execute_readonly_sql(query)
    return result_str, explanation
