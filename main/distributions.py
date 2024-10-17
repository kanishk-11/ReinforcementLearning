import torch


def categorical_distribution(logits: torch.Tensor) -> torch.distributions.Categorical:
    """Returns categorical distribution that support sample(), entropy(), and log_prob()."""
    return torch.distributions.Categorical(logits=logits)
