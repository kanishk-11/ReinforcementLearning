
import collections
import itertools
from absl import logging
import math
import multiprocessing
import queue
import signal
import sys
import threading
import time
from typing import Any, Iterable, Mapping, Text, Tuple, List
from venv import logger
import gymnasium
import torch
import tqdm
from main.abstractClasses import agent
from main.abstractClasses.action import Action
from main.abstractClasses.agent import Agent
from main.abstractClasses.learner import Learner
from main.abstractClasses.timestep import TimeStepPair
from main.checkpoint_manager import PytorchCheckpointManager
from main.logHelper import CsvWriter
from main.trackers import generate_statistics, make_default_trackers, make_learner_trackers
def get_tensorboard_log_prefix(
    env_id: str, 
    agent_name: str, 
    tag: str, 
    suffix: str
) -> str:
    """Returns the composed tensorboard log prefix,
    which is in the format {env_id}-{agent_name}-{tag}-{suffix}."""
    tb_log_prefix = f'{env_id}-{agent_name}'
    if tag is not None and tag != '':
        tb_log_prefix += f'-{tag}'
    tb_log_prefix += f'-{suffix}'
    return tb_log_prefix

def run_single_thread_training(
    num_iterations:int,
    num_training_steps:int,
    num_eval_steps:int,
    training_agent,
    training_environment:gymnasium.Env,
    evaluation_agent,
    evaluation_environment:gymnasium.Env,
    checkpoint_manager:PytorchCheckpointManager,
    csv_file:str,
    use_tensorboard:bool,
    tag:str=None,
    debug_screenshot_interval:int=0,
    folder = None
):
    writer = CsvWriter(csv_file)
    train_tensorboard_log_prefix = (
        get_tensorboard_log_prefix(training_environment.spec.id,training_agent.agent_name,tag,'train') if use_tensorboard else None
    )
    logging.info(f'debug_screenshot_interval:{debug_screenshot_interval}')
    training_trackers = make_default_trackers(train_tensorboard_log_prefix,debug_screenshot_interval,folder)
    should_evaluate = False
    evaluation_trackers = None
    if num_eval_steps>0 and evaluation_agent is not None and evaluation_environment is not None:
        should_evaluate=True
        eval_tensorboard_log_prefix = (
            get_tensorboard_log_prefix(evaluation_environment.spec.id,evaluation_agent.agent_name,tag,'evaluation') if use_tensorboard else None
        )
        evaluation_trackers = make_default_trackers(eval_tensorboard_log_prefix,debug_screenshot_interval,folder)
    for iteration in range(1, num_iterations+1):
        logging.info(f'Training Iteration No. {iteration}')
        training_statistics = run_env_steps(num_training_steps,training_agent,training_environment,training_trackers)
        checkpoint_manager.set_iteration(iteration)
        saved_checkpoint= checkpoint_manager.save()
        if saved_checkpoint:
            logging.info(f'New checkpoint at {saved_checkpoint}')
        log_output = [
            ('iteration', iteration, '%3d'),
            ('training_step', iteration*num_training_steps,'%5d'),
            ('training_episode_return', training_statistics['mean_episode_return'],'%2.2f'),
            ('training_num_episodes' , training_statistics['num_episodes'], '%3d'),
            ('train_step_rate',training_statistics['step_rate'],'%4.0f'),
            ('training_duration' , training_statistics['duration'], '%.2f')
        ]
        
        if should_evaluate is True:
            logging.info(f'Evaluating Iteration No. {iteration}')
            evaluation_statistics = run_env_steps(num_eval_steps,evaluation_agent,evaluation_environment,evaluation_trackers)
            log_output.extend([
                ('evaluation_step',iteration*num_eval_steps,'%5d'),
                ('evaluation_episode_return', evaluation_statistics['mean_episode_return'],'%2.2f'),
                ('evaluation_num_episodes' , evaluation_statistics['num_episodes'], '%3d'),
                ('eval_step_rate',evaluation_statistics['step_rate'],'%4.0f'),
                ('evaluation_duration' , evaluation_statistics['duration'], '%.2f')
            ])
        log_output_str = ', '.join(('%s: ' + f) % (n, v) for n, v, f in log_output)
        logging.info(log_output_str)
        writer.write(collections.OrderedDict((n, v) for n, v, _ in log_output))
    writer.close()
def run_env_steps(
    num_steps: int, 
    agent: Agent, 
    env: gymnasium.Env, 
    trackers: Iterable[Any]
) -> Mapping[Text, float]:
    """Run some steps and return the statistics, this could be either training, evaluation, or testing steps.

    Returns:
        A Dict contains statistics about the result.

    """
    seq = run_env_loop(agent, env)
    seq_truncated = itertools.islice(seq, num_steps)
    stats = generate_statistics(trackers, seq_truncated)
    return stats

def run_env_loop(
    agent: Agent,
    env:gymnasium.Env
)->Iterable[Tuple[gymnasium.Env, TimeStepPair, Agent, Action]]:
    if not isinstance(agent, Agent):
        raise RuntimeError('Expect agent to be an instance of abstractClasses.agent.Agent.')
    while True:
        agent.reset()
        observation,info = env.reset()
        observation = observation
        reward = 0.0
        done = loss_life = False
        first_step=True
        info = {}
        while True:
            timestep_t = TimeStepPair(
                observation=observation,
                reward=reward,
                done=done or loss_life,
                first=first_step,
                info=info,
            )
            action_t = agent.step(timestep_t)
            yield (env, timestep_t, agent, action_t)
            action_t_minus_1 = action_t
            (observation, reward, terminated,truncated, info) = env.step(action_t_minus_1)
            done = terminated or truncated
            first_step = False
            
            #Atari
            loss_life = False
            if 'loss_life' in info and info['loss_life']:
                loss_life = info['loss_life']
            
            if done:
                timestep_t = TimeStepPair(
                    observation=observation,
                    reward=reward,
                    done=True,
                    first=False,
                    info=info
                )
                unused_action = agent.step(timestep_t)
                yield env, timestep_t, agent, None
                break

def run_parallel_training_iterations(
    num_iterations:int,
    num_train_steps:int,
    num_eval_steps:int,
    learner:Learner,
    eval_agent:Agent,
    eval_env:gymnasium.Env,
    actors:List[Agent],
    actor_envs:List[gymnasium.Env],
    data_queue:multiprocessing.Queue,
    checkpoint_manager:PytorchCheckpointManager,
    csv_file:str,
    use_tensorboard:bool,
    tag:str = None,
    debug_screenshots_interval:int = 0,
    folder = None
)->None:
    iteration_count = multiprocessing.Value('i', 0)
    start_iteration_event = multiprocessing.Event()
    stop_event = multiprocessing.Event()
    log_queue = multiprocessing.SimpleQueue()
    learner_thread = threading.Thread(
        target=run_learner,
        args=(
            num_iterations,
            num_eval_steps,
            learner,
            eval_agent,
            eval_env,
            data_queue,
            log_queue,
            iteration_count,
            start_iteration_event,
            stop_event,
            checkpoint_manager,
            len(actors),
            use_tensorboard,
            tag,
            folder
        )
    )
    learner_thread.start()
    logger = threading.Thread(
        target=run_logger,
        args=(
            log_queue,
            csv_file
        )
    )
    logger.start()
    num_actors = len(actors)
    actors_tensorboard_log_prefix = [None for i in range(num_actors)]
    if use_tensorboard:
        _steps = 1 if num_actors <=8 else math.ceil(num_actors/8)
        for i in range(0,num_actors,_steps):
            actors_tensorboard_log_prefix[i] = get_tensorboard_log_prefix(actor_envs[i].spec.id,actors[i].agent_name,tag,"train")
    
    processes:List[multiprocessing.Process] = []
    for actor, actor_env, tensorboard_log_prefix in zip(actors,actor_envs,actors_tensorboard_log_prefix):
        process = multiprocessing.Process(
            target=run_actor,
            args=(
                actor,
                actor_env,
                data_queue,
                log_queue,
                num_train_steps,
                iteration_count,
                start_iteration_event,
                stop_event,
                tensorboard_log_prefix,
                debug_screenshots_interval,
                folder
            )
        )
        process.start()
        processes.append(process)
    
    for process in processes:
        process.join()
        process.close()
    logger.join()
    data_queue.close()

def run_learner(
    num_iterations:int,
    num_eval_steps:int,
    learner:Learner,
    eval_agent:Agent,
    eval_env:gymnasium.Env,
    data_queue:multiprocessing.Queue,
    log_queue:multiprocessing.SimpleQueue,
    iteration_count:multiprocessing.Value,
    start_iteration_event:multiprocessing.Event,
    stop_iteration_event:multiprocessing.Event,
    checkpoint_manager:PytorchCheckpointManager,
    num_actors:int,
    use_tensorboard:bool,
    tag:str = None,
    folder = None,
)->None:
    learner_tensorboard_log_prefixes = get_tensorboard_log_prefix(eval_env.spec.id,learner.agent_name,tag,'train') if use_tensorboard else None
    learner_trackers = make_learner_trackers(learner_tensorboard_log_prefixes,folder=folder)
    for tracker in learner_trackers:
        tracker.reset()
    should_run_evaluator = False
    evaluation_trackers = None
    
    if num_eval_steps > 0 and eval_agent is not None and eval_env is not None:
        should_run_evaluator = True
        evaluation_tracker_tensorboard_prefix = (
            get_tensorboard_log_prefix(eval_env.spec.id,eval_agent.agent_name,tag,'eval') if use_tensorboard else None
        )
        evaluation_trackers = make_default_trackers(evaluation_tracker_tensorboard_prefix,folder=folder) 
    
    for iteration in range(1,num_iterations + 1):
        logging.info(f'Training iteration {iteration}')
        logging.info(f'Starting {learner.agent_name}')
        iteration_count.value = iteration
        start_iteration_event.set()
        learner.reset()
        run_learner_loop(
            learner,
            data_queue,
            num_actors,
            learner_trackers
        )
        start_iteration_event.clear()
        checkpoint_manager.set_iteration(iteration)
        saved_checkpoint = checkpoint_manager.save()
        if saved_checkpoint:
            logging.info(f'Saved checkpoint at iteration {iteration} at {saved_checkpoint}')
        if should_run_evaluator is True:
            logging.info(f'Starting {eval_agent.agent_name}')
            evaluation_stats = run_env_steps(num_eval_steps,eval_agent,eval_env,evaluation_trackers)
            log_output = [
                ('iteration',iteration,'%3d'),
                ('role','evaluation','%3s'),
                ('step',iteration * num_eval_steps,'%5d'),
                ('episode_return',evaluation_stats['mean_episode_return'],'%2.2f'),
                ('num_episodes',evaluation_stats['num_episodes'],'%3d'),
                ('step_rate',evaluation_stats['step_rate'],'%4.0f'),
                ('duration',evaluation_stats['duration'],'%.2f')
            ]
            log_queue.put(log_output)
        time.sleep(5)
    stop_iteration_event.set()
    log_queue.put('PROCESS_DONE')

def run_learner_loop(
    learner:Learner,
    data_queue:multiprocessing.Queue,
    num_actors:int,
    learner_trackers:Iterable[Any]
):
    num_actors_done = 0
    while True:
        try:
            item = data_queue.get()
            if item == 'PROCESS_DONE':
                num_actors_done +=1
            else:
                learner.received_item_from_queue(item)
        except queue.Empty:
            pass
        except EOFError:
            pass
        if num_actors_done == num_actors:
            break
        stats_sequence = learner.step()
        if stats_sequence is not None:
            for stats in stats_sequence:
                for tracker in learner_trackers:
                    tracker.step(stats)
def run_logger(
    log_queue:multiprocessing.SimpleQueue,
    csv_file:str
):
    writer = CsvWriter(csv_file)
    while True:
        try:
            log_output = log_queue.get()
            if log_output == 'PROCESS_DONE':
                break
            log_output_str = ', '.join(('%s: ' + f) % (n, v) for n, v, f in log_output)
            logging.info(log_output_str)
            writer.write(collections.OrderedDict((n, v) for n, v, _ in log_output))
        except queue.Empty:
            pass
        except EOFError:
            pass
def run_actor(
    actor:Agent,
    actor_env:gymnasium.Env,
    data_queue:multiprocessing.Queue,
    log_queue:multiprocessing.SimpleQueue,
    num_train_steps:int,
    iteration_count:multiprocessing.Value,
    start_iteration_event:multiprocessing.Event,
    stop_iteration_event:multiprocessing.Event,
    tensorboard_log_prefix:str = None,
    debug_screenshots_interval:int = 0,
    folder=None
):
    init_absl_logging()
    handle_exit_signal()
    actor_trackers = make_default_trackers(tensorboard_log_prefix,debug_screenshots_interval,folder)
    while not stop_iteration_event.is_set():
        if not start_iteration_event.is_set():
            continue
        logging.info(f'Starting Actor:{actor.agent_name}')
        iteration = iteration_count.value
        train_stats = run_env_steps(num_train_steps,actor,actor_env,actor_trackers)
        try:
            data_queue.put('PROCESS_DONE', block=False)
        except queue.Full:
            # Queue is full, log the issue and retry
            logging.info("Data queue is full. Retrying...")
            retry_count = 0
            while retry_count < 10:  # Try up to 5 times
                time.sleep(1)  # Wait a bit before retrying
                try:
                    data_queue.put('PROCESS_DONE', block=False)
                    break  # Successfully put data in queue
                except queue.Full:
                    retry_count += 1
                    logging.info(f"Data queue still full. Retry {retry_count}/10")
            
            if retry_count == 10:
                logging.info("Failed to put data in queue after 10 attempts. Skipping.")
        if start_iteration_event.is_set():
            start_iteration_event.clear()
        log_output = [
            ('iteration',iteration,'%3d'),
            ('role',actor.agent_name,'%2s'),
            ('step',iteration * num_train_steps,'%5d'),
            ('episode_return',train_stats['mean_episode_return'],'%2.2f'),
            ('num_episodes',train_stats['num_episodes'],'%3d'),
            ('step_rate',train_stats['step_rate'],'%4.0f'),
            ('duration',train_stats['duration'],'%.2f')
        ]
        log_queue.put(log_output)
def init_absl_logging():
    """Initialize absl.logging when run the process without app.run()"""
    logging._warn_preinit_stderr = 0 
    logging.set_verbosity(logging.INFO)
    logging.use_absl_handler()
def handle_exit_signal():
    """Listen to exit signal like ctrl-c or kill from os and try to exit the process forcefully."""

    def shutdown(signal_code, frame):
        del frame
        logging.info(
            f'Received signal {signal_code}: terminating process...',
        )
        sys.exit(128 + signal_code)

    # Listen to signals to exit process.
    # signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)