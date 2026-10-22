import os

from dotenv import load_dotenv
from typing import Optional
from pydantic import BaseModel, ConfigDict

load_dotenv()


class Settings(BaseModel):
    app_name: str = "BERT Multiclass Classification API"
    description: str = "API для мультиклассовой классификации текста на основе BERT модели"
    version: str = "1.0.0"

    id2label: dict[int, str] = {0: "negative", 1: "neutral", 2: "positive"}
    # RAM во время дообучения не хватило на более длинные последовательности, увы, а у RuModernBERT-base контекст до 8к доходит
    max_length: int = 512
    device: Optional[str] = os.getenv("DEVICE") or None

    BASE_BERT_MODEL: str = os.getenv(
            "BASE_BERT_MODEL",
            "deepvk/RuModernBERT-base"
        )

    LORA_ADAPTER_WEIGHTS: str = os.getenv(
            "LORA_ADAPTER_WEIGHTS",
            "DinaYatsuk/bert-lora-sentiment"
        )

    model_config = ConfigDict(
        use_enum_values=False, extra="forbid", validate_assignment=True
    )

settings = Settings()
