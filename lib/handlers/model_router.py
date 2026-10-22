import time

from fastapi import APIRouter, Depends, HTTPException, Request

from models.bert_model import BERTClassifier
from dto import TextClassificationRequest, TextClassificationResponse

router = APIRouter(prefix="/classify", tags=["classification"])


def get_model(request: Request) -> BERTClassifier:
    model = getattr(request.app.state, "bert_model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    return model

@router.post("", response_model=TextClassificationResponse)
def classify_text(
    request: TextClassificationRequest,
    model: BERTClassifier = Depends(get_model),
):

    start_time = time.time()

    try:
        predictions = model.predict(request.text)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка классификации: {e}")

    return TextClassificationResponse(
        text=request.text,
        predictions=predictions,
        processing_time=time.time() - start_time,
    )