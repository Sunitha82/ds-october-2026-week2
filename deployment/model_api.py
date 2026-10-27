from contextlib import asynccontextmanager
from pathlib import Path
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from deployment.predictor import IncomePredictor


logger = logging.getLogger("uvicorn.error")

ARTIFACT_DIR = (
    Path(__file__).resolve().parents[1] / "assessment_outputs"
)


class IncomeRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        str_min_length=1,
        allow_inf_nan=False,
    )

    age: int = Field(ge=1, le=120)
    workclass: str
    education_num: int = Field(alias="education.num", ge=1, le=16)
    marital_status: str = Field(alias="marital.status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: float = Field(alias="capital.gain", ge=0)
    capital_loss: float = Field(alias="capital.loss", ge=0)
    hours_per_week: int = Field(alias="hours.per.week", ge=1, le=168)
    native_country: str = Field(alias="native.country")


class IncomeResponse(BaseModel):
    predicted_class: int
    income_category: str
    model_score: float
    threshold: float
    model_name: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once, before accepting requests.
    app.state.predictor = IncomePredictor(ARTIFACT_DIR)
    logger.info("Income model and preprocessor loaded.")

    yield

    # Release our reference when the service shuts down.
    del app.state.predictor


app = FastAPI(
    title="Adult Income Model Service",
    description="Runs inference using our saved census income model.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ready",
        "service": "income-model",
        "model_name": app.state.predictor.metadata["selected_model"],
    }


@app.post("/predict", response_model=IncomeResponse)
def predict(features: IncomeRequest):
    # Convert Python names back to the original dataset column names.
    person = features.model_dump(by_alias=True)
    predictor = app.state.predictor

    # Validate categories against the fitted encoder.
    encoder = predictor.preprocessor.named_transformers_["categorical"]

    for column, categories in zip(
        predictor.metadata["categorical_columns"],
        encoder.categories_,
    ):
        value = person[column]

        if value == "?":
            value = predictor.metadata["preparation"]["missing_category"]

        if value not in categories:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported category for '{column}': {value}",
            )

        person[column] = value

    try:
        return predictor.predict(person)
    except Exception:
        logger.exception("Income prediction failed.")
        raise HTTPException(
            status_code=500,
            detail="The model could not complete the prediction.",
        )