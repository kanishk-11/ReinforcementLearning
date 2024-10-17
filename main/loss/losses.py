
import logging
from venv import logger
import torch
from main.loss.common import DDQLearningExtra, LossAndExtra, QLearningExtra , get_batched_index

def q_learning_loss(q_t_minus_1:torch.Tensor,action_t_minus_1:torch.Tensor,reward:torch.Tensor,discount_t:torch.Tensor,q_dagger_t:torch.Tensor) -> LossAndExtra:
    with torch.no_grad():
        target_t_minus_1 = reward + discount_t * torch.max(q_dagger_t,dim=1)[0]
    qa_t_minus_1 = get_batched_index(q_t_minus_1,action_t_minus_1)
    td_error = target_t_minus_1 - qa_t_minus_1
    loss = 0.5 * td_error**2
    return LossAndExtra(loss,QLearningExtra(target=target_t_minus_1,td_error=td_error))

def double_q_learning_loss( 
    q_t_minus_1: torch.Tensor,
    a_t_minus_1: torch.Tensor,
    reward: torch.Tensor,
    discount_t: torch.Tensor,
    q_t_value: torch.Tensor,
    q_t_selector: torch.Tensor) -> LossAndExtra:
    
    best_action = torch.argmax(q_t_selector,dim=1)
    # logging.info(f'q_t_selector:{q_t_selector.shape},best_action.shape:{best_action.shape}')
    double_q_bootstrapped = get_batched_index(q_t_value, best_action)
    with torch.no_grad():
        target_t_minus_1 = reward + discount_t * double_q_bootstrapped
    qa_t_minus_1 = get_batched_index(q_t_minus_1,a_t_minus_1)
    td_error = target_t_minus_1 - qa_t_minus_1
    loss = 0.5 * td_error**2
    return LossAndExtra(loss,DDQLearningExtra(target=target_t_minus_1,td_error=td_error,best_action=best_action))
    
def value_loss(
    target:torch.Tensor,
    predict:torch.Tensor
):
    loss = 0.5 * torch.square(target-predict)
    if len(loss.shape)==2:
        loss = torch.mean(loss,dim=0)
    
    return LossAndExtra(loss, None)
def policy_gradient_loss(logits_t , action_t , adv_t):
    m = torch.distributions.Categorical(logits=logits_t)
    logprob_a_t = m.log_prob(action_t).view_as(adv_t)
    loss = logprob_a_t * adv_t.detach()
    
    if len(loss.shape)==2:
        loss = torch.mean(loss,dim=0)
    return LossAndExtra(loss,None)
    