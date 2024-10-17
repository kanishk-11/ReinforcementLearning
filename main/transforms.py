
import functools
from typing import Any, Callable, NamedTuple

import torch


class TransformationPair(NamedTuple):
    forward : Callable[[Any],Any]
    inverse : Callable[[Any],Any]

def identity(self, x):
    return x

def signed_hyperbolic(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + eps * x

def signed_parabolic(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    z = torch.sqrt(1 + 4 * eps * (eps + 1 + torch.abs(x))) / 2 / eps - 1 / 2 / eps
    return torch.sign(x) * (torch.square(z) - 1)
def transform_values(build_targets, *value_argnums):
	@functools.wraps(build_targets)
	def wrapped_build_targets(transformation_pair,*args,**kwargs):
		transformation_args = list(args)
		for i in value_argnums:
			transformation_args[i] = transformation_pair.inverse(transformation_args[i])
		targets = build_targets(*transformation_args, **kwargs)
		return transformation_pair.forward(targets)
	return wrapped_build_targets
IDENTITY_PAIR = TransformationPair(identity, identity)
SIGNED_HYPERBOLIC_PAIR= TransformationPair(signed_hyperbolic, signed_parabolic)