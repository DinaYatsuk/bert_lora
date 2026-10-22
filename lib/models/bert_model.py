import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Optional
import time
from config import settings
from peft import PeftModel


class BERTClassifier:
    """BERT (+LoRA) классификатор тональности: negative / neutral / positive"""

    def __init__(
        self,
        model_name: str = settings.BASE_BERT_MODEL,
        lora_adapter_weights: str = settings.LORA_ADAPTER_WEIGHTS,
        ):

        self.model_name = model_name
        self.device = settings.device or ("cuda" if torch.cuda.is_available()
                                 else "mps" if torch.backends.mps.is_available()
                                 else "cpu")
        self.lora_adapter_weights = lora_adapter_weights
        self._load_models()

    def _load_models(self) -> None:
        # Загрузка токенизатора и модели
        print(f"Загрузка модели {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(settings.id2label),
            id2label=settings.id2label,
            label2id={v: k for k, v in settings.id2label.items()},
            problem_type="single_label_classification",
            attn_implementation="sdpa",
        )

        self.model = PeftModel.from_pretrained(base_model, self.lora_adapter_weights)
        self.model.to(self.device)
        self.model.eval()
        print(f"Модель загружена на устройство: {self.device}")

    def predict(self, text: str) -> dict:
        if not isinstance(text, str) or not text.strip() or len(text) < 3:
            raise ValueError("Передайте непустой текст отзыва и не меньше 3-х символов.")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=settings.max_length,
        )
        # Перенос на нужное устройство
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)[0]

        predicted_id = int(probabilities.argmax().item())

        return {
            "label": settings.id2label[predicted_id],
            "score": round(float(probabilities[predicted_id]), 4),
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Предсказание для батча текстов

        Args:
            texts: Список текстов для классификации

        Returns:
            Список словарей с предсказаниями для каждого текста
        """
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results