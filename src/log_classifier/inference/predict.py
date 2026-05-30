import argparse
import json
import os
from typing import Any

import torch
from transformers import AutoTokenizer

from log_classifier.teacher.model import CodeBERTClassifier
from log_classifier.teacher.train_distill_student import (
    infer_keep_layers_from_state_dict,
    truncate_student_layers,
)
from log_classifier.teacher.utils import load_label_mapping


def build_text(payload: dict[str, Any]) -> str:
    if payload.get("text"):
        return str(payload["text"])

    language = str(payload.get("language", "")).strip()
    dataset = str(payload.get("dataset", "")).strip()
    user = str(payload.get("user", "")).strip()
    assistant = str(payload.get("assistant", "")).strip()

    parts = []
    if language:
        parts.append(f"language: {language}")
    if dataset:
        parts.append(f"dataset: {dataset}")
    if user:
        parts.append(f"user: {user}")
    if assistant:
        parts.append(f"assistant: {assistant}")
    return " ".join(parts).strip()


def load_model(
    checkpoint_dir: str,
    model_name: str,
    student_keep_layers: int | None,
    device: torch.device,
    dropout_prob: float = 0.1,
    pooling_mode: str = "cls_mean",
    classifier_hidden_dim: int = 512,
    multi_sample_dropout_num: int = 1,
) -> tuple[CodeBERTClassifier, Any, dict[int, str]]:
    id2label, label2id = load_label_mapping(os.path.join(checkpoint_dir, "label_mapping.json"))
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

    state_dict = torch.load(os.path.join(checkpoint_dir, "pytorch_model.bin"), map_location="cpu")
    if student_keep_layers is None:
        student_keep_layers = infer_keep_layers_from_state_dict(state_dict)

    model = CodeBERTClassifier(
        model_name=model_name,
        num_labels=len(label2id),
        dropout_prob=dropout_prob,
        pooling_mode=pooling_mode,
        classifier_hidden_dim=classifier_hidden_dim,
        multi_sample_dropout_num=multi_sample_dropout_num,
    )
    truncate_student_layers(model, student_keep_layers)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, tokenizer, id2label


@torch.inference_mode()
def predict_one(
    model: CodeBERTClassifier,
    tokenizer,
    id2label: dict[int, str],
    text: str,
    device: torch.device,
    max_length: int,
) -> dict[str, Any]:
    encoded = tokenizer(
        [text],
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    logits = model(**encoded, return_features=False)["logits"]
    probs = torch.softmax(logits, dim=-1)[0].cpu()
    pred_id = int(torch.argmax(probs).item())
    return {
        "label": id2label[pred_id],
        "confidence": float(probs[pred_id].item()),
        "scores": {id2label[i]: float(probs[i].item()) for i in sorted(id2label)},
    }


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained log-classifier checkpoint.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--model-name", default="microsoft/unixcoder-base")
    parser.add_argument("--student-keep-layers", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--text", default=None)
    parser.add_argument("--input-jsonl", default=None)
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    parser.add_argument("--pooling-mode", default="cls_mean")
    parser.add_argument("--classifier-hidden-dim", type=int, default=512)
    parser.add_argument("--multi-sample-dropout-num", type=int, default=1)
    args = parser.parse_args()

    if not args.text and not args.input_jsonl:
        parser.error("Provide either --text or --input-jsonl.")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, tokenizer, id2label = load_model(
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model_name,
        student_keep_layers=args.student_keep_layers,
        device=device,
        pooling_mode=args.pooling_mode,
        classifier_hidden_dim=args.classifier_hidden_dim,
        multi_sample_dropout_num=args.multi_sample_dropout_num,
    )

    if args.text:
        result = predict_one(model, tokenizer, id2label, args.text, device, args.max_length)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    output_f = None
    try:
        if args.output_jsonl:
            output_f = open(args.output_jsonl, "w", encoding="utf-8")
        for payload in iter_jsonl(args.input_jsonl):
            text = build_text(payload)
            result = predict_one(model, tokenizer, id2label, text, device, args.max_length)
            merged = {**payload, "prediction": result}
            line = json.dumps(merged, ensure_ascii=False)
            if output_f:
                output_f.write(line + "\n")
            else:
                print(line)
    finally:
        if output_f:
            output_f.close()
