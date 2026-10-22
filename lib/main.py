from contextlib import asynccontextmanager

import gradio as gr
from fastapi import FastAPI

from models.bert_model import BERTClassifier
from config import settings
from gradio_ui import build_demo
from handlers.model_router import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bert_model = BERTClassifier()
    yield
    app.state.bert_model = None


app = FastAPI(
    title=settings.app_name,
    description=settings.description,
    version=settings.version,
    lifespan=lifespan,
)
app.include_router(router)


@app.get("/health", tags=["service"])
def health():
    model = getattr(app.state, "bert_model", None)
    return {"status": "healthy", "model_loaded": model is not None}


def _predict(text: str):
    return app.state.bert_model.predict(text)


def _predict_batch(texts):
    return app.state.bert_model.predict_batch(texts)


app = gr.mount_gradio_app(app, build_demo(_predict, _predict_batch), path="/ui")
