from contextlib import asynccontextmanager
from typing import Any
import logging
import os

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse


logger = logging.getLogger("uvicorn.error")

MODEL_SERVICE_URL = os.getenv(
    "MODEL_SERVICE_URL",
    "http://model:8001"
).rstrip("/")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Reuse HTTP connections across requests.
    with httpx.Client(
        base_url=MODEL_SERVICE_URL,
        timeout=httpx.Timeout(15.0, connect=3.0),
        trust_env=False,
    ) as client:
        app.state.model_client = client
        yield


app = FastAPI(
    title="Adult Income Gateway",
    description="Forwards prediction requests to the model service.",
    version="1.0.0",
    lifespan=lifespan,
)


def call_model(method: str, path: str, payload=None):
    """Call the model service and handle communication errors."""
    try:
        response = app.state.model_client.request(
            method,
            path,
            json=payload,
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="The model service took too long to respond.",
        )
    except httpx.RequestError:
        logger.exception("Could not contact model service.")
        raise HTTPException(
            status_code=503,
            detail="The model service is unavailable.",
        )

    # Preserve validation errors so callers can correct their inputs.
    if response.status_code == 422:
        try:
            return JSONResponse(
                status_code=422,
                content=response.json(),
            )
        except ValueError:
            raise HTTPException(
                status_code=502,
                detail="The model service returned an invalid response.",
            )

    if response.status_code != 200:
        logger.error(
            "Model service returned HTTP %s",
            response.status_code,
        )
        raise HTTPException(
            status_code=502,
            detail="The model service could not complete the request.",
        )

    try:
        result = response.json()
    except ValueError:
        raise HTTPException(
            status_code=502,
            detail="The model service returned invalid JSON.",
        )

    if not isinstance(result, dict):
        raise HTTPException(
            status_code=502,
            detail="The model service returned an unexpected response.",
        )

    return result


@app.get("/health")
def health():
    result = call_model("GET", "/health")

    if isinstance(result, JSONResponse) or result.get("status") != "ready":
        raise HTTPException(
            status_code=503,
            detail="The model service is not ready.",
        )

    return {
        "status": "ready",
        "service": "income-gateway",
        "model_service": result,
    }


@app.post("/predict")
def predict(person: dict[str, Any]):
    return call_model("POST", "/predict", payload=person)