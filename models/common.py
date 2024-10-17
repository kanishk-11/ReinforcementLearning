import logging

import torch



def get_state_and_action_dim(environment,FLAGS):
    logging.info('Environment: %s', FLAGS.environment_name)
    logging.info('Action spec: %s', environment.action_space.n)
    logging.info('Observation spec: %s', environment.observation_space.shape)
    state_dim = environment.observation_space.shape
    action_dim = environment.action_space.n
    return state_dim,action_dim

def disable_autograd(network:torch.nn.Module):
    for parameter in network.parameters():
        parameter.requires_grad = False
def numpy_to_tensor(array,device,dtype):
    return torch.from_numpy(array).to(device=device,dtype=dtype)