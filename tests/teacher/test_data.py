import json
import os

import pytest

from log_classifier.teacher.data import ClassificationDataset, build_label_mapping, read_dataset


@pytest.fixture
def dummy_jsonl(tmp_path):
    data = [
        {"id": 1, "text": "user: hello", "label_text": "greeting"},
        {"id": 2, "text": "user: hi there", "label_text": "greeting"},
        {"id": 3, "text": "user: bye", "label_text": "farewell"},
    ]
    path = os.path.join(tmp_path, "dummy.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")
    return path


def test_jsonl_loading(dummy_jsonl):
    data = read_dataset(dummy_jsonl)
    assert len(data) == 3
    assert data[0]["id"] == 1
    assert data[0]["label_text"] == "greeting"


def test_label_mapping(dummy_jsonl):
    data = read_dataset(dummy_jsonl)
    label2id, id2label = build_label_mapping(data)
    assert "greeting" in label2id
    assert "farewell" in label2id
    assert len(label2id) == 2
    assert id2label[label2id["greeting"]] == "greeting"


def test_classification_dataset(dummy_jsonl):
    data = read_dataset(dummy_jsonl)
    label2id, _ = build_label_mapping(data)
    
    dataset = ClassificationDataset(data, label2id)
    assert len(dataset) == 3
    
    item = dataset[0]
    assert "id" in item
    assert "text" in item
    assert "label" in item
    assert "label_text" in item
    
    assert item["label"] == label2id["greeting"]
