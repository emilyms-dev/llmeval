"""FastAPI server that exposes llmeval storage over HTTP for the dashboard."""

from llmeval.server.api import create_app

__all__ = ["create_app"]
