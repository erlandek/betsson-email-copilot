"""Short-term memory: the SQLite checkpointer that lets a run pause and resume.

LangGraph stores the graph state after every step. Our state holds Pydantic
objects, and LangGraph 1.x warns before deserialising types it has not been
told about (a checkpoint database could otherwise be used to load arbitrary
classes). So we allow exactly the classes in assistant.schemas.
"""

import inspect
import sqlite3
from enum import Enum
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel

from assistant import schemas


def _allowed_types() -> list[tuple[str, str]]:
    return [
        (cls.__module__, cls.__name__)
        for _, cls in inspect.getmembers(schemas, inspect.isclass)
        if cls.__module__ == schemas.__name__ and issubclass(cls, (BaseModel, Enum))
    ]


def serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=_allowed_types())


def sqlite_checkpointer(path: Path) -> SqliteSaver:
    path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False), serde=serializer())
