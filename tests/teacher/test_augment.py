import pytest

from log_classifier.teacher.augment import TextCodeAugmenter


def test_augment_deterministic():
    text = "user: What is this?\nassistant: ```python\ndef foo():\n    return 1\n```"
    aug1 = TextCodeAugmenter(seed=42)
    res1 = aug1.augment(text, n=3)

    aug2 = TextCodeAugmenter(seed=42)
    res2 = aug2.augment(text, n=3)

    assert res1 == res2


def test_augment_never_empty():
    aug = TextCodeAugmenter()
    text = "user: hi"
    for _ in range(50):
        res = aug.augment(text, n=1)[0]
        assert len(res.strip()) > 0


def test_remove_code_comments():
    aug = TextCodeAugmenter()
    text = "def foo():\n    # this is a comment\n    return 1 // inline comment\n/* block */"
    res = aug.remove_code_comments(text)
    assert "# this is a comment" not in res
    assert "// inline comment" not in res
    assert "/* block */" not in res
    assert "def foo():" in res


def test_remove_role_markers():
    aug = TextCodeAugmenter()
    text = "user: Hello!\nassistant: Hi there."
    res = aug.remove_role_markers(text)
    assert "user:" not in res
    assert "assistant:" not in res
    assert "Hello!" in res
    assert "Hi there." in res
