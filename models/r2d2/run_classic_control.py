
import copy
import logging
import multiprocessing
import os
from os.path import dirname
import logging
import sys
from absl import app
dir_name = dirname(__file__)
main_file = dirname(dirname(dirname(__file__)))
print(main_file)
sys.path.append(main_file)
import numpy as np
import torch

from main import checkpoint_manager
from main.ReplayUtils.replayBuffers import PrioritizedReplay
from main.actors import R2D2EpsilonGreedyActor
from main.checkpoint_manager import PytorchCheckpointManager
from main.gym import gym_environment
from main.networks.value_networks import R2D2DQN
from models import common_loops
from models.r2d2.r2d2 import Actor, R2D2Learner, R2D2Transition
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from absl import flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'environment_name',
    'LunarLander-v2',
    'Classic control tasks name like CartPole-v1, LunarLander-v2, MountainCar-v0, Acrobot-v1.',
    )
flags.DEFINE_integer('num_actors', 8, 'Number of actors')
flags.DEFINE_integer('replay_size', 10000, 'Replay capacity')
flags.DEFINE_integer('min_replay_size',1000, 'Minimum replay size before starting to train')
flags.DEFINE_bool('clip_gradients', True, 'flags to clip the gradients')
flags.DEFINE_float('max_gradient_norm',0.5, 'max gradient for gradient clipping')
flags.DEFINE_float('learning_rate',0.0005,'learning rate')
flags.DEFINE_float('adam_epsilon', 0.001,'epsilon for adam optimizer')
flags.DEFINE_float('discount' , 0.997,'Discount Rate')
flags.DEFINE_integer('unroll_length', 15, 'Sequence of transitions to unroll before add to replay.')
flags.DEFINE_integer(
    'burn_in',
    0,
    'Sequence of transitions used to pass RNN before actual learning.'
    'The effective length of unrolls will be burn_in + unroll_length, '
    'two consecutive unrolls will overlap on burn_in steps.',
)
flags.DEFINE_integer('batch_size', 32, 'Batch size for learning.')

flags.DEFINE_float('priority_exponent', 0.9, 'Priority exponent used in prioritized replay.')
flags.DEFINE_float('importance_sampling_exponent', 0.6, 'Importance sampling exponent value.')
flags.DEFINE_bool('normalize_weights', True, 'Normalize sampling weights in prioritized replay.')
flags.DEFINE_float('priority_eta', 0.9, 'Priority eta to mix the max and mean absolute TD errors.')
flags.DEFINE_float('rescale_epsilon', 0.001, 'Epsilon used in the invertible value rescaling for n-step targets.')
flags.DEFINE_integer('n_step', 5, 'TD n-step bootstrap.')
flags.DEFINE_integer('num_iterations', 2, 'Number of iterations to run.')
flags.DEFINE_integer('num_train_steps', int(5e5), 'Number of training env steps to run per iteration, per actor.')
flags.DEFINE_integer('num_eval_steps', int(2e4), 'Number of evaluation env steps to run per iteration.')
flags.DEFINE_integer(
    'target_net_update_interval',
    400,
    'The interval (meassured in Q network updates) to update target Q networks.',
)
flags.DEFINE_integer('actor_update_interval', 100, 'The frequency (measured in actor steps) to update actor local Q network.')
flags.DEFINE_float('eval_exploration_epsilon', 0.00001, 'Fixed exploration rate in e-greedy policy for evaluation.')
flags.DEFINE_integer('seed', 1, 'Runtime seed.')
flags.DEFINE_bool('use_tensorboard', True, 'Use Tensorboard to monitor statistics, default on.')
flags.DEFINE_bool('actors_on_gpu', True, 'Run actors on GPU, default on.')
flags.DEFINE_integer(
    'debug_screenshots_interval',
    0,
    'Take screenshots every N episodes and log to Tensorboard, default 0 no screenshots.',
)
flags.DEFINE_string('tag', '', 'Add tag to Tensorboard log file.')
flags.DEFINE_string('results_csv_path', './logs/r2d2_classic_results.csv', 'Path for CSV log file.')
flags.DEFINE_string('checkpoint_dir', './checkpoints', 'Path for checkpoint directory.')

def main(argv):
    del argv
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Using device: %s' % device)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
    random_state = np.random.RandomState(FLAGS.seed)
    def build_env():
        return gym_environment.get_classic_environment(
            env_name = FLAGS.environment_name,
            seed = random_state.randint(1,2**14)
        )
    train_env = build_env()
    eval_env = build_env()
    action_dim = train_env.action_space.__dict__.__getitem__('n')
    state_dim = train_env.observation_space.shape[0]
    logging.info(f'environment name:{FLAGS.environment_name}')
    logging.info(f'observation space dimension:{train_env.observation_space.shape}')
    logging.info(f'action space dimension:{action_dim}')
    network = R2D2DQN(state_dim=state_dim, action_dim=action_dim)
    optimizer = torch.optim.AdamW(network.parameters(), lr=FLAGS.learning_rate, eps=FLAGS.adam_epsilon)
    observation , _ = train_env.reset()
    output = network(torch.from_numpy(observation[None,None,...]).float(),torch.zeros(1,1).long(),torch.zeros(1,1),network.get_initial_hidden_state(1))
    assert output.q_values.shape == (1, 1, action_dim)
    assert len(output.hidden_state) == 2
    importance_sampling_exponent = FLAGS.importance_sampling_exponent

    def importance_sampling_exponent_schedule(x):
        return importance_sampling_exponent

    replay = PrioritizedReplay(
        size=FLAGS.replay_size,
        structure=R2D2Transition(None,None,None,None,None,None,None,None),
        priority_exponent=FLAGS.priority_exponent,
        importance_sampling_exponent=importance_sampling_exponent_schedule,
        normalize_weights=FLAGS.normalize_weights,
        random_state=random_state,
        time_major=True,
    )
    data_queue = multiprocessing.Queue(maxsize=FLAGS.num_actors * 2)

    manager = multiprocessing.Manager()

    shared_params = manager.dict({'network': None})
    learner_agent = R2D2Learner(
        network=network,
        optimizer=optimizer,
        replay_buffer=replay,
        min_replay_size=FLAGS.min_replay_size,
        target_network_update_interval=FLAGS.target_net_update_interval,
        discount=FLAGS.discount,
        burn_in=FLAGS.burn_in,
        priority_eta=FLAGS.priority_eta,
        rescale_epsilon=FLAGS.rescale_epsilon,
        batch_size=FLAGS.batch_size,
        n_steps=FLAGS.n_step,
        clip_gradient=FLAGS.clip_gradients,
        max_gradient_norm=FLAGS.max_gradient_norm,
        device=device,
        shared_parameters=shared_params,
    )
    actor_envs = [build_env() for _ in range(FLAGS.num_actors)]

    actor_devices = ['cpu'] * FLAGS.num_actors
    if torch.cuda.is_available() and FLAGS.actors_on_gpu:
        num_gpus = torch.cuda.device_count()
        actor_devices = [torch.device(f'cuda:{i % num_gpus}') for i in range(FLAGS.num_actors)]

    # Rank 0 is the most explorative actor, while rank N-1 is the most exploitative actor.
    # Each actor has it's own network with different weights.
    actors = [
        Actor(
            rank=i,
            data_queue=data_queue,
            network=copy.deepcopy(network),
            random_state=np.random.RandomState(FLAGS.seed + int(i)),  # pylint: disable=no-member
            num_actors=FLAGS.num_actors,
            action_dim=action_dim,
            unroll_length=FLAGS.unroll_length,
            burn_in=FLAGS.burn_in,
            actor_update_interval=FLAGS.actor_update_interval,
            device=actor_devices[i],
            shared_parameters=shared_params,
        )
        for i in range(FLAGS.num_actors)
    ]
    # Create evaluation agent instance
    eval_agent = R2D2EpsilonGreedyActor(
        network=network,
        exploration_epsilon=FLAGS.eval_exploration_epsilon,
        random_state=random_state,
        device=device,
    )

    # Setup checkpoint.
    checkpoint_manager = PytorchCheckpointManager(environment_name=FLAGS.environment_name, agent_name='R2D2', save_dir=FLAGS.checkpoint_dir)
    checkpoint_manager.register_pair(('network', network))

    # Run parallel training N iterations.
    common_loops.run_parallel_training_iterations(
        num_iterations=FLAGS.num_iterations,
        num_train_steps=FLAGS.num_train_steps,
        num_eval_steps=FLAGS.num_eval_steps,
        learner=learner_agent,
        eval_agent=eval_agent,
        eval_env=eval_env,
        actors=actors,
        actor_envs=actor_envs,
        data_queue=data_queue,
        checkpoint_manager=checkpoint_manager,
        csv_file=FLAGS.results_csv_path,
        use_tensorboard=FLAGS.use_tensorboard,
        tag=FLAGS.tag,
        debug_screenshots_interval=FLAGS.debug_screenshots_interval,
    )
if __name__ == '__main__':
    # Set multiprocessing start mode
    multiprocessing.set_start_method('spawn')
    app.run(main)