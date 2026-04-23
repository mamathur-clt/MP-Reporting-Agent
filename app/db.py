"""
Shared Databricks connection factory.

Supports two auth modes, selected automatically:

1. **Service Principal (Databricks Apps)** — when ``DATABRICKS_CLIENT_ID``
   and ``DATABRICKS_CLIENT_SECRET`` are present (injected by the platform),
   the Databricks SDK handles OAuth token generation and refresh.
2. **Personal Access Token (local dev)** — falls back to ``DATABRICKS_TOKEN``
   from ``.env`` when no service-principal credentials are found.

Every data module (``data.py``, ``seo_data.py``, ``finance_data.py``,
``paid_search_data.py``, ``analyst_tools.py``) and the monitor/bot should
import ``get_connection`` from here instead of rolling their own.
"""

import logging
import os
import time

import certifi
from databricks import sql as databricks_sql
from dotenv import load_dotenv

load_dotenv(override=True)

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

log = logging.getLogger(__name__)

_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH", "")

# Detect which auth mode to use at import time.
_USE_SP = bool(
    os.getenv("DATABRICKS_CLIENT_ID") and os.getenv("DATABRICKS_CLIENT_SECRET")
)

if _USE_SP:
    from databricks.sdk.core import Config as _DatabricksConfig

    _cfg = _DatabricksConfig()
    _HOSTNAME = _cfg.host.replace("https://", "").strip("/") if _cfg.host else ""
else:
    _HOSTNAME = os.getenv("DATABRICKS_HOST", "").replace("https://", "").strip("/")
    _TOKEN = os.getenv("DATABRICKS_TOKEN", "")


_MAX_RETRIES = 2
_RETRY_DELAY_S = 1.0


def get_connection():
    """Return a fresh ``databricks.sql`` connection.

    Retries once on transient / stale-connection errors so that idle
    Databricks Apps deployments recover automatically.
    """
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            if _USE_SP:
                return databricks_sql.connect(
                    server_hostname=_HOSTNAME,
                    http_path=_HTTP_PATH,
                    credentials_provider=lambda: _cfg.authenticate,
                )
            return databricks_sql.connect(
                server_hostname=_HOSTNAME,
                http_path=_HTTP_PATH,
                access_token=_TOKEN,
            )
        except Exception as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                log.warning(
                    "Databricks connection attempt %d failed (%s), retrying…",
                    attempt + 1,
                    exc,
                )
                time.sleep(_RETRY_DELAY_S)
    raise last_exc  # type: ignore[misc]
