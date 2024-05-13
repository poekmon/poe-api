import asyncio
import logging
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from poe_api.config import CONFIG, load_config
from poe_api.exceptions import SelfDefinedException
from poe_api.logger import get_log_config, setup_logger
from poe_api.middlewares.asgi_logger.middleware import AccessLoggerMiddleware
from poe_api.response import handle_exception_response
from poe_api.router import api_router

setup_logger()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    AccessLoggerMiddleware, format="%(client_addr)s | %(request_line)s | %(status_code)s | %(M)s ms", logger=logging.getLogger("poe_api.access")
)

app.include_router(api_router)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return handle_exception_response(exc)


@app.exception_handler(SelfDefinedException)
async def self_defined_exception_handler(request, exc):
    return handle_exception_response(exc)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=CONFIG.host,
        port=CONFIG.port,
        proxy_headers=True,
        forwarded_allow_ips="*",
        log_config=get_log_config(),
    )
