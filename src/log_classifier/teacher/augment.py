import random
import re


class TextCodeAugmenter:
    """Small deterministic text/code perturbation helper for robustness tests."""

    def __init__(self, seed: int | None = None):
        self.random = random.Random(seed)

    def remove_role_markers(self, text: str) -> str:
        return re.sub(r"\b(user|assistant)\s*:\s*", "", text, flags=re.IGNORECASE)

    def remove_markdown_noise(self, text: str) -> str:
        text = re.sub(r"```[a-zA-Z0-9_+-]*", "", text)
        return text.replace("```", "")

    def remove_code_comments(self, text: str) -> str:
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
        text = re.sub(r"(?m)^\s*#.*$", "", text)
        text = re.sub(r"(?m)//.*$", "", text)
        return text

    def normalize_whitespace(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def truncate_assistant_explanation(self, text: str) -> str:
        marker = "assistant:"
        lower_text = text.lower()
        marker_index = lower_text.find(marker)
        if marker_index < 0:
            return text

        head = text[: marker_index + len(marker)]
        tail = text[marker_index + len(marker) :].strip()
        words = tail.split()
        if len(words) <= 32:
            return text
        return f"{head} {' '.join(words[:32])}"

    def add_harmless_noise(self, text: str) -> str:
        insertions = ["", " ", "\n", "  "]
        return text + self.random.choice(insertions)

    def augment(self, text: str, n: int = 1) -> list[str]:
        transforms = [
            self.remove_role_markers,
            self.remove_markdown_noise,
            self.remove_code_comments,
            self.normalize_whitespace,
            self.truncate_assistant_explanation,
            self.add_harmless_noise,
        ]

        outputs = []
        for _ in range(int(n)):
            transformed = text
            for transform in self.random.sample(transforms, k=self.random.randint(1, len(transforms))):
                candidate = transform(transformed)
                if candidate.strip():
                    transformed = candidate
            outputs.append(transformed if transformed.strip() else text)
        return outputs
