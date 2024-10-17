import sys
from os.path import dirname



dir_name = dirname(__file__)
main_file = dirname(dirname(dirname(__file__)))
print(main_file)
sys.path.append(main_file)
from models.ddqn_prioritised_replay.ddqn_prioritised_replay import DDQNPER
from main.checkpoint_manager import PytorchCheckpointManager
from models import common_loops
from main.actors import EpsilonGreedyActor
from main.ReplayUtils.transactionAccumulators import TransitionAccumulator
from main.scheduler import LinearScheduler
from main.ReplayUtils.common import ReplayData
from main.ReplayUtils.replayBuffers import PrioritizedReplay, UniformReplay
from main.ReplayUtils.transition import Transition
import logging
from absl import flags
import numpy as np
import torch
from absl import app
import torch.backends
import torch.backends.cudnn
from main.gym import gym_environment
FLAG = flags.FLAGS
from main.networks.value_networks import DQNNet, DQNNetAA
flags.DEFINE_string(
    'env_name',
    'LunarLander-v2',
    'classic control environments that have a linear state[ CartPole-v1, LunarLander-v2, MountainCar-v0, Acrobot-v1 ]'
)
flags.DEFINE_integer(
    'replay_size',
    100000,
    'Replay Buffer Size'
)
flags.DEFINE_integer(
    'batch_size',
    512,
    'Batch Size'
)
flags.DEFINE_integer(
    'target_network_update_interval',
    100,
    'Target Network Update Interval'
)
flags.DEFINE_integer(
    'learning_interval',
    5,
    'Learning Interval'
)
flags.DEFINE_integer(
    'min_replay_size',
    10000,
    'Minimum replay size before learning'
)
flags.DEFINE_integer(
    'num_iterations',
    4,
    'Number of Iterations'
)
flags.DEFINE_integer(
    'num_training_steps',
    int(5e5),
    'Number of Training Steps per iteration'
)
flags.DEFINE_integer(
    'num_eval_steps',
    int(1e5),
    'Number of Evaluation Steps per iteration'
)
flags.DEFINE_integer(
    'seed',
    1,
    'Random Seed'
)
flags.DEFINE_integer(
    'debug_screenshots_interval',
    1000,
    'Take screenshots every N episodes and log to Tensorboard, default 0 no screenshots.',
)
flags.DEFINE_bool(
    'clip_gradient',
    False,
    'Clip Gradient(True/False) Default:False'
)
flags.DEFINE_bool(
    'use_tensorboard',
    True,
    'Use Tensorboard(True/False) Default:True'
)
flags.DEFINE_bool(
    'use_gpu',
    True,
    'Use GPU(True/False) Default:True'
)
flags.DEFINE_float(
    'max_grad_norm',
    0.5,
    'Max Gradient while clipping'
)
flags.DEFINE_float(
    'exploration_epsilon_begin',
    1.0,
    'Exploration Epsilon Beginning value'
)
flags.DEFINE_float(
    'exploration_epsilon_end',
    0.001,
    'Exploration Epsilon Ending value'
)
flags.DEFINE_float(
    'exploration_epsilon_decay_step',
    1000000,
    'Total steps to decay value of the exploration rate.')

flags.DEFINE_float(
    'learning_rate',
    0.0005,
    'Learning Rate For updating weights'
)
flags.DEFINE_float(
    'discount',
    0.99,
    'Discount Factor'
)
flags.DEFINE_float(
    'eval_exploration_epsilon',
    0.0001,
    'Evaluation Exploration Epsilon'
)
flags.DEFINE_string(
    'checkpoint_dir',
    dir_name+'/checkpoints',
    'Checkpoint Directory'
)
flags.DEFINE_string(
    'tag', 
    '', 
    'Add tag to Tensorboard log file.'
)
flags.DEFINE_string(
    'results_csv_path', 
    dir_name+ '/logs/dqn_classic_results.csv', 
    'Path for CSV log file.'
)
flags.DEFINE_float('priority_exponent', 0.6, 'Priority exponent used in prioritized replay.')
flags.DEFINE_float('importance_sampling_exponent_begin_value', 0.4, 'Importance sampling exponent begin value.')
flags.DEFINE_float('importance_sampling_exponent_end_value', 1.0, 'Importance sampling exponent end value after decay.')
flags.DEFINE_boolean('normalize_weights', True, 'normalize weights')
def main(argv):
    del argv
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    logging.info(f'running on device: {device}')
    np.random.seed(FLAG.seed)
    torch.manual_seed(FLAG.seed)
    random_state = np.random.RandomState(FLAG.seed)    
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
    def build_env():
        return gym_environment.get_classic_environment(
            env_name = FLAG.env_name,
            seed = random_state.randint(1,2**14)
        )
    training_environment = build_env()
    evaluation_environment = build_env()
    action_dim = training_environment.action_space.__dict__.__getitem__('n')
    state_dim = training_environment.observation_space.shape[0]
    logging.info(f'environment name:{FLAG.env_name}')
    logging.info(f'observation space dimension:{training_environment.observation_space.shape}')
    logging.info(f'action space dimension:{action_dim}')
    
    network = DQNNetAA(state_dimension=state_dim,action_dimension=action_dim)
    optimizer = torch.optim.AdamW(network.parameters(),lr = FLAG.learning_rate)
    observation = training_environment.reset()
    q_values = network(torch.from_numpy(observation[0]).float().unsqueeze(dim=0)).q_values
    assert q_values.shape == (1,action_dim)
    
    exploration_rate_scheduler = LinearScheduler(
        beginning_t=FLAG.min_replay_size,
        beginning_value=FLAG.exploration_epsilon_begin,
        end_value = FLAG.exploration_epsilon_end,
        decay_steps = FLAG.exploration_epsilon_decay_step,
    )
    importance_sampling_exponent_schedule = LinearScheduler(
        beginning_t=int(FLAG.min_replay_size),
        decay_steps=(3* int(FLAG.num_training_steps)),
        beginning_value=FLAG.importance_sampling_exponent_begin_value,
        end_value=FLAG.importance_sampling_exponent_end_value
    )
    replay = PrioritizedReplay(size=FLAG.replay_size,structure=Transition(None,None,None,None,None),priority_exponent=FLAG.priority_exponent,
        importance_sampling_exponent=importance_sampling_exponent_schedule,
        normalize_weights=FLAG.normalize_weights,
        random_state=random_state,)
    training_agent = DDQNPER(
        network=network,
        optimizer=optimizer,
        transition_accumulator=TransitionAccumulator(),
        replay_buffer=replay,
        device=device,
        random_state=random_state,
        exploration_rate_lambda=exploration_rate_scheduler,
        clip_gradient=FLAG.clip_gradient,
        batch_size=FLAG.batch_size,
        min_replay_size=FLAG.min_replay_size,
        learning_interval=FLAG.learning_interval,
        target_network_update_interval=FLAG.target_network_update_interval,
        max_gradient_norm=FLAG.max_grad_norm,
        action_space_dimension=action_dim,
        discount=FLAG.discount
    )
    evaluation_agent = EpsilonGreedyActor(
        network=network,
        exploration_epsilon=FLAG.eval_exploration_epsilon,
        random_state=random_state,
        device=device,
        name='DDQN-PER_Greedy'
    )
    checkpoint = PytorchCheckpointManager(environment_name=FLAG.env_name, agent_name='DDQN', save_dir=FLAG.checkpoint_dir)
    checkpoint.register_pair(('network', network))

    # Run the training and evaluation for N iterations.
    common_loops.run_single_thread_training(
        num_iterations=FLAG.num_iterations,
        num_training_steps=FLAG.num_training_steps,
        num_eval_steps=FLAG.num_eval_steps,
        training_agent=training_agent,
        training_environment=training_environment,
        evaluation_agent=evaluation_agent,
        evaluation_environment=evaluation_environment,
        checkpoint_manager=checkpoint,
        csv_file=FLAG.results_csv_path,
        use_tensorboard=FLAG.use_tensorboard,
        tag=FLAG.tag,
        debug_screenshot_interval=FLAG.debug_screenshots_interval,
        folder=dir_name
    )
    
if __name__ == '__main__':
    app.run(main)



