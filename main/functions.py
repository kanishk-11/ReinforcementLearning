from ast import arg
import functools
from typing import List
from cv2 import transform
import numpy as np
import torch

from main.loss.common import LossAndExtra, QLearningExtra, get_batched_index
from main.transforms import IDENTITY_PAIR, transform_values

def n_step_bellman_target(
	r_t: torch.Tensor,
	done: torch.Tensor,
	q_t: torch.Tensor,
	gamma: float,
	n_steps: int,
) -> torch.Tensor:
	r"""Computes n-step Bellman targets.

	See section 2.3 of R2D2 paper (which does not mention the logic around end of
	episode).

	Args:
	  rewards: This is r_t in the equations below. Should be non-discounted, non-summed,
		shape [T, B] tensor.
	  done: This is done_t in the equations below. done_t should be true
		if the episode is done just after
		experimenting reward r_t, shape [T, B] tensor.
	  q_t: This is Q_target(s_{t+1}, a*) (where a* is an action chosen by the caller),
		shape [T, B] tensor.
	  gamma: Exponential RL discounting.
	  n_steps: The number of steps to look ahead for computing the Bellman targets.

	Returns:
	  y_t targets as <float32>[time, batch_size] tensor.
	  When n_steps=1, this is just:

	  $$r_t + gamma * (1 - done_t) * Q_{target}(s_{t+1}, a^*)$$

	  In the general case, this is:

	  $$(\sum_{i=0}^{n-1} \gamma ^ {i} * notdone_{t, i-1} * r_{t + i}) +
		\gamma ^ n * notdone_{t, n-1} * Q_{target}(s_{t + n}, a^*) $$

	  where notdone_{t,i} is defined as:

	  $$notdone_{t,i} = \prod_{k=0}^{k=i}(1 - done_{t+k})$$

	  The last n_step-1 targets cannot be computed with n_step returns, since we
	  run out of Q_{target}(s_{t+n}). Instead, they will use n_steps-1, .., 1 step
	  returns. For those last targets, the last Q_{target}(s_{t}, a^*) is re-used
	  multiple times.
	"""

	# We append n_steps - 1 times the last q_target. They are divided by gamma **
	# k to correct for the fact that they are at a 'fake' indices, and will
	# therefore end up being multiplied back by gamma ** k in the loop below.
	# We prepend 0s that will be discarded at the first iteration below.
	bellman_target = torch.concat(
		[torch.zeros_like(q_t[0:1]), q_t] + [q_t[-1:] / gamma**k for k in range(1, n_steps)], dim=0
	)
	# Pad with n_steps 0s. They will be used to compute the last n_steps-1
	# targets (having 0 values is important).
	done = torch.concat([done] + [torch.zeros_like(done[0:1])] * n_steps, dim=0)
	rewards = torch.concat([r_t] + [torch.zeros_like(r_t[0:1])] * n_steps, dim=0)
	# Iteratively build the n_steps targets. After the i-th iteration (1-based),
	# bellman_target is effectively the i-step returns.
	for _ in range(n_steps):
		rewards = rewards[:-1]
		done = done[:-1]
		bellman_target = rewards + gamma * (1.0 - done.float()) * bellman_target[1:]

	return bellman_target
def calculate_distributed_priorities_from_td_error(td_error: torch.Tensor, eta: float) -> np.ndarray:
	td_errors = torch.clone(td_error).detach()
	absolute_td_errors = torch.abs(td_errors)
	priorities = eta * torch.max(absolute_td_errors,dim=0)[0] + (1-eta) * torch.mean(absolute_td_errors,dim=0)
	priorities = torch.clamp(priorities, min = 0.00001, max = 1000)
	priorities = priorities.cpu().numpy()
	return priorities

def get_actors_exploration_rate(n: int) -> List[float]:
	assert 1 <= n
	return np.power(0.4, np.linspace(1.0, 8.0, num=n)).flatten().tolist()
def general_off_policy_returns_from_action_values(
	q_t:torch.Tensor,
	action_t:torch.Tensor,
    reward_t:torch.Tensor,
    discount_t:torch.Tensor,
    c_t:torch.Tensor,
    pi_t:torch.Tensor,
) -> torch.Tensor:
	"""
	
	"""
	exp_q_t = (pi_t * q_t).sum(axis=-1)
	q_a_t = get_batched_index(q_t,action_t)[:-1,...]
	c_t = c_t[:-1,...]
	return general_off_policy_returns_from_q_and_v(q_a_t,exp_q_t,reward_t,discount_t,c_t)
def general_off_policy_returns_from_q_and_v(
	q_t:torch.Tensor,
    v_t:torch.Tensor,
    reward_t:torch.Tensor,
    discount_t:torch.Tensor,
    c_t:torch.Tensor,
):
    g = reward_t[-1] + discount_t[-1]* v_t[-1]
    returns = [g]
    for i in reversed(range(q_t.shape[0])): #T
        g = reward_t[i] + discount_t[i] * (v_t[i] - c_t[i] * q_t[i] + c_t[i] * g)
        returns.insert(0,g)
    return torch.stack(returns,dim=0).detach()
transformed_general_off_policy_returns_from_action_values = transform_values(
	general_off_policy_returns_from_action_values,0
)

def transformed_retrace(
	q_t_minus_1:torch.Tensor,
	q_t:torch.Tensor,
	action_t_minus_1:torch.Tensor,
	action_t:torch.Tensor,
	reward_t:torch.Tensor,
	discount_t:torch.Tensor,
	pi_t : torch.Tensor,
	mu_t : torch.Tensor,
	lambda_:float,
	epsilon : float = 1e-8,
	transformation_pair = IDENTITY_PAIR
):
	pi_action_t = get_batched_index(pi_t,action_t)
	c_t = torch.minimum(torch.tensor(1.0),pi_action_t/(mu_t + epsilon)) * lambda_
	with torch.no_grad():
		target_t_minus_1 = transformed_general_off_policy_returns_from_action_values(transformation_pair,q_t,action_t,reward_t,discount_t,c_t,pi_t)
	q_a_t_minus_1 = get_batched_index(q_t_minus_1,action_t_minus_1)
	td_error = (target_t_minus_1-q_a_t_minus_1)
	loss = 0.5 * (td_error**2)
	return LossAndExtra(
		loss=loss,
		extraInformation=QLearningExtra(
		target=target_t_minus_1,
		td_error=td_error
		)
	)
def get_ngu_policy_betas_and_gammas(
	num_policies:int,
	beta:float = 0.3,
	gamma_minimum:float = 0.99,
	gamma_maximum:float = 0.997
):
    return get_ngu_betas(num_policies,beta),get_ngu_gammas(num_policies,gamma_minimum,gamma_maximum)
def get_ngu_betas(
	num_policies:int,
    beta:float = 0.3
):
	def sigmoid(x):
		return 1 / (1 + np.exp(-x))

	results = []
	for i in range(num_policies):
		if i == 0:
			results.append(0.0)
		elif i == num_policies - 1:
			results.append(beta)
		else:
			beta = beta * sigmoid(((2 * i - (num_policies - 2)) / (num_policies - 2)))
			results.append(beta)

	return results
def get_ngu_gammas(
	num_policies:int,
    gamma_minimum:float = 0.99,
    gamma_maximum:float = 0.997
):
    results = []
    for i in range(num_policies):
        _numerator = (num_policies-1-i)*np.log(1-gamma_maximum) + i * np.log(1-gamma_minimum)
        _gamma_i = 1 - np.exp(_numerator/(num_policies-1))
        results.append(_gamma_i)
    return results

