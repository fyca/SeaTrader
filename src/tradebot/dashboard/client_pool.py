"""
Global Alpaca client pool for dashboard endpoints.

Creates clients once at app startup and reuses them across all requests.
This prevents memory bloat from creating hundreds of new client objects.
"""

from typing import Optional

# Global client pool (initialized by create_app)
_pooled_clients: Optional[object] = None


def init_client_pool(clients) -> None:
    """Initialize the global client pool. Called by create_app."""
    global _pooled_clients
    _pooled_clients = clients


def get_pooled_clients():
    """Get the pooled Alpaca clients. Raises if not initialized."""
    global _pooled_clients
    if _pooled_clients is None:
        raise RuntimeError("Client pool not initialized. Did create_app run?")
    return _pooled_clients
