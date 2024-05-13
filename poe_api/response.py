import json
import logging
import typing
from typing import Optional, Any, Generic, TypeVar, Dict

import httpx
from fastapi import Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ValidationError
from starlette.background import BackgroundTask
from starlette.exceptions import HTTPException as StarletteHTTPException

from poe_api.exceptions import SelfDefinedException

logger = logging.getLogger(__name__)


def response(code, content):
    return Response(content, code)


def handle_exception_response(e: Exception):
    if isinstance(e, ValidationError):
        return response(400, f"errors.validationError")
    elif isinstance(e, SelfDefinedException):
        return response(e.code, f"{e.reason}: {e.message}")
    # logger.error(f"Unhandled exception: {e}")
    return response(500, "errors.internal")
