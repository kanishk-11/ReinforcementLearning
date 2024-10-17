from absl import app
from absl import flags
from absl import logging
import os
from os.path import dirname
import sys

dir_name = dirname(__file__)
main_file = dirname(dirname(dirname(__file__)))
print(main_file)
sys.path.append(main_file)
from models import common_loops
from main.ReplayUtils.common import compress, decompress
from main.ReplayUtils.replayBuffers import PrioritizedReplay
from main.checkpoint_manager import PytorchCheckpointManager
from main.networks.curiosity_networks import NGUEmbedding, RNDConv
from main.networks.value_networks import NGUConv, NGUNetworkInputs
from models.ngu.ngu import NGUActor, NGULearner, NGUTransition
from main.actors import NGUEpsilonGreedyActor
from main.gym import gym_environment


import multiprocessing
import numpy as np
import torch
import copy
from grokfast_pytorch import GrokFastAdamW
FLAGS = flags.FLAGS
flags.DEFINE_string(
    'environment_name',
    'Pong',
    'The name of the environment'
)
flags.DEFINE_integer(
    'environment_height',
    84,
    'The height of the environment'
)
flags.DEFINE_integer(
    'environment_width',
    84,
    'The width of the environment'
)
flags.DEFINE_integer(
    'environment_frame_skip',
    4,
    'The number of frames to skip'
)
flags.DEFINE_integer(
    'environment_frame_stack',
    1,
    'The number of frames to stack'
)
flags.DEFINE_integer(
    'num_actors',
    8,
    'The number of actors'
)
flags.DEFINE_bool(
    'compress_state',
    True,
    'Compress state images when stored'
)
flags.DEFINE_integer(
    'replay_size',
    20000,
    'maximum replay size'
)
flags.DEFINE_integer(
    'minimum_replay_size',
    1000,
    'Minimum replay size before learning'
)
flags.DEFINE_bool(
    'clip_gradients',
    True,
    'Clip gradients'
)
flags.DEFINE_float(
    'max_gradient_norm',
    40.0,
    'max gradient norm'
)
flags.DEFINE_float(
    'learning_rate',
    0.0001,
    'learning rate'
)
flags.DEFINE_float(
    'intrinsic_learning_rate',
    0.0005,
    'intrinsic learning rate[For embedding and RND]'
)
flags.DEFINE_float('extrinsic_discount', 0.997, 'Extrinsic reward discount rate.')
flags.DEFINE_float('intrinsic_discount', 0.99, 'Intrinsic reward discount rate.')
flags.DEFINE_float('adam_epsilon', 0.0001, 'Epsilon for adam.')
flags.DEFINE_integer('unroll_length', 80, 'Sequence of transitions to unroll before add to replay.')
flags.DEFINE_integer(
    'burn_in',
    40,
    'Sequence of transitions used to pass RNN before actual learning.'
    'The effective length of unrolls will be burn_in + unroll_length, '
    'two consecutive unrolls will overlap on burn_in steps.',
)
flags.DEFINE_integer('batch_size', 32, 'Batch size for learning.')

flags.DEFINE_float('policy_beta', 0.3, 'Scalar for the intrinsic reward scale.')
flags.DEFINE_integer('num_policies', 32, 'Number of directed policies to learn, scaled by intrinsic reward scale beta.')

flags.DEFINE_integer('episodic_memory_size', 30000, 'Maximum size of episodic memory.')  # 30000
flags.DEFINE_bool(
    'reset_episodic_memory',
    True,
    'Reset the episodic_memory on every episode, only applicable to actors, default on.'
    'From NGU Paper on MontezumaRevenge, Instead of resetting the memory after every episode, we do it after a small number of '
    'consecutive episodes, which we call a meta-episode. This structure plays an important role when the'
    'agent faces irreversible choices.',
)
flags.DEFINE_integer('num_neighbors', 10, 'Number of K-nearest neighbors.')
flags.DEFINE_float('kernel_epsilon', 0.0001, 'K-nearest neighbors kernel epsilon.')
flags.DEFINE_float('cluster_distance', 0.008, 'K-nearest neighbors custer distance.')
flags.DEFINE_float('max_similarity', 8.0, 'K-nearest neighbors custer distance.')

flags.DEFINE_float('retrace_lambda', 0.95, 'Lambda coefficient for retrace.')
flags.DEFINE_bool('transformed_retrace', True, 'Transformed retrace loss, default on.')

flags.DEFINE_float('priority_exponent', 0.9, 'Priority exponent used in prioritized replay.')
flags.DEFINE_float('importance_sampling_exponent', 0.6, 'Importance sampling exponent value.')
flags.DEFINE_bool('normalize_weights', True, 'Normalize sampling weights in prioritized replay.')
flags.DEFINE_float('priority_eta', 0.9, 'Priority eta to mix the max and mean absolute TD errors.')

flags.DEFINE_integer('num_iterations', 100, 'Number of iterations to run.')
flags.DEFINE_integer(
    'num_train_steps', int(5e5), 'Number of training steps (environment steps or frames) to run per iteration, per actor.'
)
flags.DEFINE_integer(
    'num_eval_steps', int(2e4), 'Number of evaluation steps (environment steps or frames) to run per iteration.'
)
flags.DEFINE_integer('max_episode_steps', 108000, 'Maximum steps (before frame skip) per episode.')
flags.DEFINE_integer(
    'target_network_update_interval',
    1500,
    'The interval (meassured in Q network updates) to update target Q networks.',
)
flags.DEFINE_integer('actor_update_interval', 100, 'The frequency (measured in actor steps) to update actor local Q network.')
flags.DEFINE_float('eval_exploration_epsilon', 0.0001, 'Fixed exploration rate in e-greedy policy for evaluation.')
flags.DEFINE_integer('seed', 1, 'Runtime seed.')
flags.DEFINE_bool('use_tensorboard', True, 'Use Tensorboard to monitor statistics, default on.')
flags.DEFINE_bool('actors_on_gpu', True, 'Run actors on GPU, default on.')
flags.DEFINE_integer(
    'debug_screenshots_interval',
    0,
    'Take screenshots every N episodes and log to Tensorboard, default 0 no screenshots.',
)
flags.DEFINE_string('tag', '', 'Add tag to Tensorboard log file.')
flags.DEFINE_string('results_csv_path', './logs/ngu_atari_results.csv', 'Path for CSV log file.')
flags.DEFINE_string('checkpoint_dir', './checkpoints', 'Path for checkpoint directory.')

flags.register_validator('environment_frame_stack', lambda x: x == 1)
torch.autograd.set_detect_anomaly(True)
def main(argv):
    del argv
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Runs NGU agent on {device}')
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    random_state = np.random.RandomState(FLAGS.seed)
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
            sticky_action=False,
            clip_reward=False,
        )
    eval_env = environment_builder()
    action_dim = eval_env.action_space.__dict__.__getitem__('n')
    state_dim = eval_env.observation_space.shape
    
    logging.info(f'environment name:{FLAGS.environment_name}')
    logging.info(f'observation space dimension:{state_dim}')
    logging.info(f'action space dimension:{action_dim}')
    network = NGUConv(state_dim, action_dim,FLAGS.num_policies)
    optimizer = GrokFastAdamW(network.parameters(True),lr=FLAGS.learning_rate,eps =FLAGS.adam_epsilon)

    observation , _ = eval_env.reset()
    RND_target_net = RNDConv(state_dim=state_dim,is_target=True)
    RND_predictor_net = RNDConv(state_dim=state_dim,is_target=False)
    
    NGU_embedding_net = NGUEmbedding(state_dim=state_dim,action_dim=action_dim,embedding_size=256)
    intrinsic_embedding_optimizer = GrokFastAdamW(
        NGU_embedding_net.parameters(True),
        lr = FLAGS.intrinsic_learning_rate,
        eps=FLAGS.adam_epsilon
    )
    intrinsic_rnd_optimizer = GrokFastAdamW(
        RND_predictor_net.parameters(True),
        lr = FLAGS.intrinsic_learning_rate,
        eps=FLAGS.adam_epsilon
    )
    inp = NGUNetworkInputs(
        state_t=torch.from_numpy(observation[None,None,...]).float(),
        action_t_minus_1=torch.zeros(1,1).long(),
        extrinsic_reward_t=torch.zeros(1, 1).float(),
        intrinsic_reward_t=torch.zeros(1, 1).float(),
        policy_index_t=torch.zeros(1, 1).long(),
        hidden_state=network.get_initial_hidden_state(1),
    )
    network_output = network(inp)
    assert network_output.q_values.shape == (1, 1, action_dim)
    assert len(network_output.hidden_state) == 2
    importance_sampling_exponent = FLAGS.importance_sampling_exponent

    def importance_sampling_exponent_schedule(x):
        return importance_sampling_exponent
    if FLAGS.compress_state:

        def encoder(transition):
            return transition._replace(
                state_t=compress(transition.state_t),
            )

        def decoder(transition):
            return transition._replace(
                state_t=decompress(transition.state_t),
            )

    else:
        encoder = None
        decoder = None
    replay = PrioritizedReplay(
        size=FLAGS.replay_size,
        structure=NGUTransition(None,None,None,None,None,None,None,None,None,None,None,None,None),
        priority_exponent=FLAGS.priority_exponent,
        importance_sampling_exponent=importance_sampling_exponent_schedule,
        normalize_weights=FLAGS.normalize_weights,
        random_state=random_state,
        time_major=True,
        encoder=encoder,
        decoder=decoder
    )
    data_queue = multiprocessing.Queue(maxsize=FLAGS.num_actors * 2)

    manager = multiprocessing.Manager()

    shared_params = manager.dict(
        {
            'network': None,
            'embedding_network': None,
            'rnd_predictor_network': None,
            'timestamp': None,
        }
    )
    learner = NGULearner(
        network=network,
        optimizer=optimizer,
        embedding_network=NGU_embedding_net,
        RND_predictor_network=RND_predictor_net,
        RND_target_network=RND_target_net,
        intrinsic_embedding_optimizer=intrinsic_embedding_optimizer,
        intrinsic_rnd_optimizer=intrinsic_rnd_optimizer,
        replay_buffer=replay,
        target_network_update_interval=FLAGS.target_network_update_interval,
        minimum_replay_size=FLAGS.minimum_replay_size,
        batch_size=FLAGS.batch_size,
        unroll_length=FLAGS.unroll_length,
        burn_in=FLAGS.burn_in,
        retrace_lambda=FLAGS.retrace_lambda,
        transformed_retrace=FLAGS.transformed_retrace,
        priority_eta=FLAGS.priority_eta,
        clip_gradients=FLAGS.clip_gradients,
        max_gradient_norm=FLAGS.max_gradient_norm,
        device=device,
        shared_parameters=shared_params
    )
    actor_envs = [environment_builder() for _ in range(FLAGS.num_actors)]

    actor_devices = ['cpu'] * FLAGS.num_actors
    if torch.cuda.is_available() and FLAGS.actors_on_gpu:
        num_gpus = torch.cuda.device_count()
        actor_devices = [torch.device(f'cuda:{i % num_gpus}') for i in range(FLAGS.num_actors)]
    actors = [NGUActor(
        rank=i,
        data_queue=data_queue,
        network=copy.deepcopy(network),
        RND_predictor_network=copy.deepcopy(RND_predictor_net),
        RND_target_network=copy.deepcopy(RND_target_net),
        embedding_network=copy.deepcopy(NGU_embedding_net),
        random_state=np.random.RandomState(FLAGS.seed + int(i)),
        extrinsic_discount=FLAGS.extrinsic_discount,
        intrinsic_discount=FLAGS.intrinsic_discount,
        policy_beta=FLAGS.policy_beta,
        num_actors=FLAGS.num_actors,
        action_dim=action_dim,
        unroll_length=FLAGS.unroll_length,
        burn_in=FLAGS.burn_in,
        num_policies=FLAGS.num_policies,
        reset_episodic_memory=FLAGS.reset_episodic_memory,
        num_neighbors=FLAGS.num_neighbors,
        cluster_distance=FLAGS.cluster_distance,
        kernel_epsilon=FLAGS.kernel_epsilon,
        maximum_similarity=FLAGS.max_similarity,
        actor_update_interval=FLAGS.actor_update_interval,
        device=actor_devices[i],
        shared_parameters=shared_params,
        episodic_memory_size=FLAGS.episodic_memory_size,
    ) for i in range(FLAGS.num_actors)]
    eval_agent = NGUEpsilonGreedyActor(
        network=network,
        rnd_predictor_network=RND_predictor_net,
        rnd_target_network=RND_target_net,
        embedding_network=NGU_embedding_net,
        exploration_epsilon = FLAGS.eval_exploration_epsilon,
        episodic_memory_size=FLAGS.episodic_memory_size,
        num_neighbors = FLAGS.num_neighbors,
        kernel_epsilon=FLAGS.kernel_epsilon,
        cluster_distance = FLAGS.cluster_distance,
        maximum_similarity=FLAGS.max_similarity,
        random_state=random_state,
        device=device
    )
    checkpoint_manager = PytorchCheckpointManager(environment_name=FLAGS.environment_name, agent_name='R2D2', save_dir=FLAGS.checkpoint_dir)
    checkpoint_manager.register_pair(('network', network))
    checkpoint_manager.register_pair(('rnd_target_network', RND_target_net))
    checkpoint_manager.register_pair(('rnd_predictor_network', RND_predictor_net))
    checkpoint_manager.register_pair(('embedding_network', NGU_embedding_net))
    common_loops.run_parallel_training_iterations(
        num_iterations=FLAGS.num_iterations,
        num_train_steps=FLAGS.num_train_steps,
        num_eval_steps=FLAGS.num_eval_steps,
        learner=learner,
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
        folder=dir_name
    )
if __name__ == '__main__':
    # Set multiprocessing start mode
    multiprocessing.set_start_method('spawn')
    app.run(main)