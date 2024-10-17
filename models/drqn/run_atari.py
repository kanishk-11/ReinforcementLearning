
import sys
from os.path import dirname

import gymnasium
import gymnasium.envs.registration

dir_name = dirname(__file__)
main_file = dirname(dirname(dirname(__file__)))
print(main_file)
sys.path.append(main_file)
from models.ddqn.ddqn import DDQN
from models.drqn.drqn import DRQN

import logging
from absl import flags

import numpy as np
import torch
from main.gym import gym_environment

from main.ReplayUtils.common import compress, decompress
from main.ReplayUtils.replayBuffers import UniformReplay
from main.ReplayUtils.transactionAccumulators import TransitionAccumulator
from main.ReplayUtils.transition import Transition
from main.actors import DRQNEpsilonGreedyActor, EpsilonGreedyActor
from main.checkpoint_manager import PytorchCheckpointManager
from main.networks.value_networks import DQNConvNet, DRQNConv
from main.scheduler import LinearScheduler
from models import common_loops
from models.dqn.dqn import DQN
import models.common as common

FLAGS = flags.FLAGS
flags.DEFINE_string('environment_name', 'Breakout', 'Atari name without NoFrameskip and version, like Breakout, Pong, Seaquest.')
flags.DEFINE_integer('environment_height', 84, 'Environment frame screen height.')
flags.DEFINE_integer('environment_width', 84, 'Environment frame screen width.')
flags.DEFINE_integer('environment_frame_skip', 4, 'Number of frames to skip.')
flags.DEFINE_integer('environment_frame_stack', 4, 'Number of frames to stack.')
flags.DEFINE_bool('compress_state', True, 'Compress state images when store in experience replay.')
flags.DEFINE_integer('replay_capacity', 200000, 'Maximum replay size.')
flags.DEFINE_integer('min_replay_size', 10000, 'Minimum replay size before learning starts.')
flags.DEFINE_integer('batch_size', 8, 'Sample batch size when updating the neural network.')
flags.DEFINE_bool('clip_grad', False, 'Clip gradients, default off.')
flags.DEFINE_float('max_grad_norm', 10.0, 'Max gradients norm when do gradients clip.')
flags.DEFINE_float('exploration_epsilon_begin_value', 1.0, 'Begin value of the exploration rate in e-greedy policy.')
flags.DEFINE_float('exploration_epsilon_end_value', 0.0001, 'End (decayed) value of the exploration rate in e-greedy policy.')
flags.DEFINE_float(
    'exploration_epsilon_decay_step',
    int(1e6),
    'Total steps (after frame skip) to decay value of the exploration rate.',
)
flags.DEFINE_float('eval_exploration_epsilon', 0.00001, 'Fixed exploration rate in e-greedy policy for evaluation.')
flags.DEFINE_float('learning_rate', 0.00025, 'Learning rate.')
flags.DEFINE_float('discount', 0.99, 'Discount rate.')
flags.DEFINE_integer('num_iterations', 100, 'Number of iterations to run.')
flags.DEFINE_integer(
    'num_train_steps', int(5e5), 'Number of training steps (environment steps or frames) to run per iteration.'
)
flags.DEFINE_integer(
    'num_eval_steps', int(1e5), 'Number of evaluation steps (environment steps or frames) to run per iteration.'
)
flags.DEFINE_integer('max_episode_steps', 108000, 'Maximum steps (before frame skip) per episode.')
flags.DEFINE_integer('learn_interval', 4, 'The frequency (measured in agent steps) to update parameters.')
flags.DEFINE_integer(
    'target_net_update_interval',
    2000 ,
    'The frequency (measured in number of Q network parameter updates) to update target networks.',
)
flags.DEFINE_integer('seed', 1, 'Runtime seed.')
flags.DEFINE_bool('use_tensorboard', True, 'Use Tensorboard to monitor statistics, default on.')
flags.DEFINE_bool('actors_on_gpu', True, 'Run actors on GPU, default on.')
flags.DEFINE_integer(
    'debug_screenshots_interval',
    100,
    'Take screenshots every N episodes and log to Tensorboard, default 0 no screenshots.',
)
flags.DEFINE_string('tag', '', 'Add tag to Tensorboard log file.')
flags.DEFINE_string('results_csv_path', dir_name+'/logs/ddqn_atari_results.csv', 'Path for CSV log file.')
flags.DEFINE_string('checkpoint_dir', dir_name+'/checkpoints', 'Path for checkpoint directory.')
flags.DEFINE_integer('unroll_length', 10, 'Unroll length')
from absl import app

def main(argv):
    """Trains DQN agent on Atari."""
    del argv
    runtime_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Runs DDQN agent on {runtime_device}')
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    random_state = np.random.RandomState(FLAGS.seed)  # pylint: disable=no-member
    def environment_builder():
            return gym_environment.get_atari_environment(
                env_name=FLAGS.environment_name,
                frame_height=FLAGS.environment_height,
                frame_width=FLAGS.environment_width,
                frame_skip=FLAGS.environment_frame_skip,
                frame_stack=FLAGS.environment_frame_stack,
                max_episode_steps=FLAGS.max_episode_steps,
                seed=random_state.randint(1, 2**10),
                noop_max=30,
                terminal_on_life_loss=True,
            )
    train_env = environment_builder()
    eval_env = environment_builder()

    state_dim , action_dim = common.get_state_and_action_dim(train_env,FLAGS)

    # Test environment and state shape.
    obs,info = train_env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (FLAGS.environment_frame_stack, FLAGS.environment_height, FLAGS.environment_width)

    network = DRQNConv(state_dim=state_dim, action_dim=action_dim)
    optimizer = torch.optim.AdamW(network.parameters(), lr=FLAGS.learning_rate)

    # Test network input and output
    logging.info(torch.from_numpy(obs[None,None, ...]).shape)
    q = network(torch.from_numpy(obs[None,None, ...])).q_values
    assert q.shape == (1,1, action_dim)

    # Create e-greedy exploration epsilon schedule
    exploration_epsilon_schedule = LinearScheduler(
        beginning_t=int(FLAGS.min_replay_size),
        decay_steps=int(FLAGS.exploration_epsilon_decay_step),
        beginning_value=FLAGS.exploration_epsilon_begin_value,
        end_value=FLAGS.exploration_epsilon_end_value,
    )

    # Create transition replay
    if FLAGS.compress_state:

        def encoder(transition):
            return transition._replace(
                state_t_minus_1=compress(transition.state_t_minus_1),
                state_t=compress(transition.state_t),
            )

        def decoder(transition):
            return transition._replace(
                state_t_minus_1=decompress(transition.state_t_minus_1),
                state_t=decompress(transition.state_t),
            )

    else:
        encoder = None
        decoder = None

    replay = UniformReplay(
        size=FLAGS.replay_capacity,
        structure=Transition(None,None,None,None,None),
        random_state=random_state,
        encoder=encoder,
        decoder=decoder,
    )

    # Create DQN agent instance
    train_agent = DRQN(
        network=network,
        optimizer=optimizer,
        transition_accumulator=TransitionAccumulator(),
        replay_buffer=replay,
        exploration_rate_lambda=exploration_epsilon_schedule,
        batch_size=FLAGS.batch_size,
        min_replay_size=FLAGS.min_replay_size,
        learning_interval=FLAGS.learn_interval,
        target_network_update_interval=FLAGS.target_net_update_interval,
        discount=FLAGS.discount,
        clip_gradient=FLAGS.clip_grad,
        max_gradient_norm=FLAGS.max_grad_norm,
        action_space_dimension=action_dim,
        random_state=random_state,
        device=runtime_device,
        unroll_length=FLAGS.unroll_length,
    )

    # Create evaluation agent instance
    eval_agent = DRQNEpsilonGreedyActor(
        network=network,
        exploration_epsilon=FLAGS.eval_exploration_epsilon,
        random_state=random_state,
        device=runtime_device,
        name='DRQN-greedy',
    )

    # Setup checkpoint.
    checkpoint = PytorchCheckpointManager(environment_name=FLAGS.environment_name, agent_name='DDQN', save_dir=FLAGS.checkpoint_dir)
    checkpoint.register_pair(('network', network))

    # Run the training and evaluation for N iterations.
    common_loops.run_single_thread_training(
        num_iterations=FLAGS.num_iterations,
        num_training_steps=FLAGS.num_train_steps,
        num_eval_steps=FLAGS.num_eval_steps,
        training_agent=train_agent,
        training_environment=train_env,
        evaluation_agent=eval_agent,
        evaluation_environment=eval_env,
        checkpoint_manager=checkpoint,
        csv_file=FLAGS.results_csv_path,
        use_tensorboard=FLAGS.use_tensorboard,
        tag=FLAGS.tag,
        debug_screenshot_interval=FLAGS.debug_screenshots_interval,
        folder=dir_name
    )


if __name__ == '__main__':
    app.run(main)