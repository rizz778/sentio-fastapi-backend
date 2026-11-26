"""
Node Backend Clients
====================
Modular clients for communicating with the Sentio Node.js backend.
"""

from clients.node.base import BaseNodeClient
from clients.node.session import SessionClient, session_client
from clients.node.memory import MemoryClient, memory_client

# Backward compatibility alias
NodeBackendClient = SessionClient
node_client = session_client

__all__ = [
    "BaseNodeClient",
    "SessionClient",
    "session_client",
    "MemoryClient",
    "memory_client",
    "NodeBackendClient",
    "node_client",
]
