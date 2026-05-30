import torch

from log_classifier.teacher.losses import supervised_contrastive_loss, symmetric_kl_loss, weighted_cross_entropy


def test_supervised_contrastive_loss_with_positives():
    features = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0]
    ])
    # L2 normalize
    features = torch.nn.functional.normalize(features, p=2, dim=1)
    labels = torch.tensor([0, 0, 1, 1])
    
    loss = supervised_contrastive_loss(features, labels)
    assert loss.item() >= 0
    assert not torch.isnan(loss)


def test_supervised_contrastive_loss_no_positives():
    features = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.707, 0.707]
    ])
    features = torch.nn.functional.normalize(features, p=2, dim=1)
    labels = torch.tensor([0, 1, 2])  # No two samples have same label
    
    loss = supervised_contrastive_loss(features, labels)
    # Should not crash, should return 0.0
    assert loss.item() == 0.0


def test_weighted_cross_entropy():
    logits = torch.tensor([
        [2.0, -1.0],
        [0.0, 3.0]
    ])
    labels = torch.tensor([0, 1])
    weights = torch.tensor([1.0, 2.0])
    
    loss = weighted_cross_entropy(logits, labels, weights)
    assert loss.item() > 0
    assert not torch.isnan(loss)


def test_symmetric_kl_loss():
    logits_a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    logits_b = torch.tensor([[0.5, 0.5], [0.1, 0.9]])
    
    loss = symmetric_kl_loss(logits_a, logits_b)
    assert loss.item() >= 0
    assert not torch.isnan(loss)
