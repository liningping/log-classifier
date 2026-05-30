import torch
import torch.nn.functional as F


def weighted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    return F.cross_entropy(logits, labels, weight=class_weights)


def symmetric_kl_loss(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    temperature = float(temperature)
    log_prob_a = F.log_softmax(logits_a / temperature, dim=-1)
    log_prob_b = F.log_softmax(logits_b / temperature, dim=-1)
    prob_a = log_prob_a.exp()
    prob_b = log_prob_b.exp()

    kl_ab = F.kl_div(log_prob_a, prob_b, reduction="batchmean")
    kl_ba = F.kl_div(log_prob_b, prob_a, reduction="batchmean")
    return 0.5 * (kl_ab + kl_ba) * (temperature**2)


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError("features must have shape [batch_size, feature_dim]")

    device = features.device
    batch_size = features.size(0)
    if batch_size <= 1:
        return features.new_tensor(0.0)

    features = F.normalize(features, dim=1)
    similarity = torch.matmul(features, features.T) / float(temperature)

    logits_mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
    positive_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & logits_mask

    if not positive_mask.any():
        return features.new_tensor(0.0)

    similarity = similarity - similarity.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(similarity) * logits_mask
    log_prob = similarity - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))

    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1)[valid] / positive_count[valid]
    return -mean_log_prob_pos.mean()
