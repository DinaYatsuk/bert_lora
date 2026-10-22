import time
from typing import Callable, Dict, List, Optional

import gradio as gr

PredictFn = Callable[[str], Dict]
PredictBatchFn = Callable[[List[str]], List[Dict]]

EXAMPLES = [
    ["Отличный товар, всем рекомендую! Доставили на день раньше."],
    ["Ужасное качество, развалилось через неделю. Деньги на ветер."],
    ["Обычная вещь, качество среднее, ничего особенного."],
]


def build_demo(
    predict_fn: PredictFn,
    predict_batch_fn: Optional[PredictBatchFn] = None,
) -> gr.Blocks:
    def classify(text: str):
        start = time.time()
        try:
            prediction = predict_fn(text)
        except Exception as e:
            raise gr.Error(str(e))
        return {prediction["label"]: prediction["score"]}, round(time.time() - start, 4)

    def classify_many(raw: str):
        texts = [line.strip() for line in raw.splitlines() if line.strip()]
        if not texts:
            raise gr.Error("Введи хотя бы один текст.")
        try:
            if predict_batch_fn is not None:
                results = predict_batch_fn(texts)
            else:
                results = [predict_fn(t) for t in texts]
        except Exception as e:
            raise gr.Error(str(e))
        return [[text, p["label"], p["score"]] for text, p in zip(texts, results)]

    with gr.Blocks(title="Sentiment classifier") as demo:
        gr.Markdown(
            "# Классификатор тональности\n"
            "RuModernBERT-base + LoRA: `negative` / `neutral` / `positive`"
        )

        with gr.Tab("Один текст"):
            text = gr.Textbox(label="Текст отзыва", lines=5, placeholder="Введите текст...")
            with gr.Row():
                submit = gr.Button("Классифицировать", variant="primary")
                gr.ClearButton(components=[text])
            label = gr.Label(label="Результат")
            elapsed = gr.Number(label="Время обработки, с", interactive=False)
            gr.Examples(examples=EXAMPLES, inputs=text)

            submit.click(classify, inputs=text, outputs=[label, elapsed])
            text.submit(classify, inputs=text, outputs=[label, elapsed])

        with gr.Tab("Несколько текстов"):
            raw = gr.Textbox(label="Тексты (по одному на строку)", lines=8)
            run = gr.Button("Классифицировать всё", variant="primary")
            table = gr.Dataframe(
                headers=["text", "label", "score"],
                datatype=["str", "str", "number"],
                label="Результаты",
                interactive=False,
            )
            run.click(classify_many, inputs=raw, outputs=table)

    return demo