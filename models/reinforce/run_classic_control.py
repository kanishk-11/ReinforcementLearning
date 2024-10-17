
import sys
from os.path import dirname

main_file = dirname(dirname(dirname(__file__)))
print(main_file)
sys.path.append(main_file)

from main.networks.policy_networks import ActorMLP, CriticMLP
from models.reinforce.reinforce import ReinforceWithBaseline
from main.checkpoint_manager import PytorchCheckpointManager
from models import common_loops
from main.actors import PolicyGreedyActor
from main.ReplayUtils.transactionAccumulators import TransitionAccumulator
import logging
from absl import flags
import numpy as np
import torch
from absl import app
import torch.backends
import torch.backends.cudnn
from main.gym import gym_environment
FLAG = flags.FLAGS
from grokfast_pytorch import GrokFastAdamW

FLAGS = flags.FLAGS
flags.DEFINE_string('environment_name', 'Acrobot-v1', '')
flags.DEFINE_bool('normalize_returns', False, 'Normalize episode returns, default off.')
flags.DEFINE_bool('clip_gradients', False, 'Clip gradients, default off.')
flags.DEFINE_float('max_gradient_norm', 40.0, 'Max gradients norm when do gradients clip.')
flags.DEFINE_float('learning_rate', 0.0005, 'Learning rate for policy network.')
flags.DEFINE_float('value_learning_rate', 0.0005, 'Learning rate for value network.')
flags.DEFINE_float('discount', 0.99, 'Discount rate.')
flags.DEFINE_integer('num_iterations', 2, 'Number of iterations to run.')
flags.DEFINE_integer('num_train_steps', int(5e5), 'Number of training env steps to run per iteration.')
flags.DEFINE_integer('num_eval_steps', int(2e4), 'Number of evaluation env steps to run per iteration.')
flags.DEFINE_integer('seed', 1, 'Runtime seed.')
flags.DEFINE_bool('use_tensorboard', True, 'Use Tensorboard to monitor statistics, default on.')
flags.DEFINE_bool('actors_on_gpu', True, 'Run actors on GPU, default on.')
flags.DEFINE_integer(
    'debug_screenshots_interval',
    0,
    'Take screenshots every N episodes and log to Tensorboard, default 0 no screenshots.',
)
flags.DEFINE_string(
    'checkpoint_dir',
    './models/reinforce/checkpoints',
    'Checkpoint Directory'
)
flags.DEFINE_string(
    'tag', 
    '', 
    'Add tag to Tensorboard log file.'
)
flags.DEFINE_string(
    'results_csv_path', 
    '../models/dqn/logs/dqn_classic_results.csv', 
    'Path for CSV log file.'
)

def main(argv):
    del argv
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    logging.info(f'running on device: {device}')
    np.random.seed(FLAG.seed)
    torch.manual_seed(FLAG.seed)
    random_state = np.random.RandomState(FLAG.seed)    
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    def build_env():
        return gym_environment.get_classic_environment(
            env_name = FLAG.environment_name,
            seed = random_state.randint(1,2**14)
        )
    training_environment = build_env()
    evaluation_environment = build_env()
    action_dim = training_environment.action_space.__dict__.__getitem__('n')
    state_dim = training_environment.observation_space.shape[0]
    logging.info(f'environment name:{FLAG.environment_name}')
    logging.info(f'observation space dimension:{training_environment.observation_space.shape}')
    logging.info(f'action space dimension:{action_dim}')
    
    
    policy_network = ActorMLP(input_dim=state_dim,output_dim=action_dim)
    value_network = CriticMLP(input_dim=state_dim,output_dim=action_dim)
    policy_optimiser = GrokFastAdamW(policy_network.parameters(),lr = FLAG.learning_rate)
    value_optimiser = GrokFastAdamW(value_network.parameters(),lr=FLAG.value_learning_rate)
    
    training_agent = ReinforceWithBaseline(
        policy_network=policy_network,
        policy_optimizer=policy_optimiser,
        value_network=value_network,
        baseline_optimizer=value_optimiser,
        discount=FLAG.discount,
        normalize_returns=FLAG.normalize_returns,
        clip_gradients=FLAG.clip_gradients,
        max_gradient_norm=FLAG.max_gradient_norm,
        device=device,
        transition_accumulator=TransitionAccumulator()
    )
    
    eval_agent = PolicyGreedyActor(
        network=policy_network,
        device=device,
        name='REINFORCE-BASELINE-greedy',
    )
    checkpoint = PytorchCheckpointManager(environment_name=FLAG.environment_name, agent_name='REINFORCE', save_dir=FLAG.checkpoint_dir)
    checkpoint.register_pair(('policy_network', policy_network))
    checkpoint.register_pair(('value_network',value_network))

    # Run the training and evaluation for N iterations.
    common_loops.run_single_thread_training(
        num_iterations=FLAG.num_iterations,
        num_training_steps=FLAG.num_train_steps,
        num_eval_steps=FLAG.num_eval_steps,
        training_agent=training_agent,
        training_environment=training_environment,
        evaluation_agent=eval_agent,
        evaluation_environment=evaluation_environment,
        checkpoint_manager=checkpoint,
        csv_file=FLAG.results_csv_path,
        use_tensorboard=FLAG.use_tensorboard,
        tag=FLAG.tag,
        debug_screenshot_interval=FLAG.debug_screenshots_interval,
        folder=dirname(__file__)
    )
    
if __name__ == '__main__':
    app.run(main)



