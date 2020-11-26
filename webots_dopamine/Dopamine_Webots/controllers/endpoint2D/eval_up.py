# coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The entry point for running an agent on an Atari 2600 domain.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



from absl import app
from absl import flags
# from agents.ddpg import ddpg_agent
from agents.ddpg import ddpg_agent_s
from agents.dqn import dqn_agent
from agents.implicit_quantile import implicit_quantile_agent
from agents.rainbow import rainbow_agent
import time
import os
import json
import run_experiment
from pipe import open_read_pipe, read_from_pipe
from gym import spaces
import tensorflow as tf
import numpy as np


flags.DEFINE_bool('debug_mode', True,
                  'If set to true, the agent will output in-episode statistics '
                  'to Tensorboard. Disabled by default as this results in '
                  'slower training.')
flags.DEFINE_string('agent_name', None,
                    'Name of the agent. Must be one of '
                    '(dqn, rainbow, implicit_quantile)')
flags.DEFINE_string('base_dir', None,
                    'Base directory to host all required sub-directories.')
flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files (e.g.'
    '"dopamine/agents/dqn/dqn.gin").')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "DQNAgent.epsilon_train=0.1",'
    '      "create_environment.game_name="Pong"").')
flags.DEFINE_string(
    'schedule', 'continuous_train_and_eval',
    'The schedule with which to run the experiment and choose an appropriate '
    'Runner. Supported choices are '
    '{continuous_train, continuous_train_and_eval}.')

FLAGS = flags.FLAGS

space_path = "/tmp/space_out1.pipe"

def create_agent(sess, environment, summary_writer=None):
  """Creates a DQN agent.

  Args:
    sess: A `tf.Session` object for running associated ops.
    environment: An Atari 2600 Gym environment.
    summary_writer: A Tensorflow summary writer to pass to the agent
      for in-agent training statistics in Tensorboard.

  Returns:
    agent: An RL agent.

  Raises:
    ValueError: If `agent_name` is not in supported list.
  """
  s = open_read_pipe(space_path)
  space = read_from_pipe(s)
  # print("space ", space)
  if not FLAGS.debug_mode:
    summary_writer = None
  if FLAGS.agent_name == 'ddpg':
    os.close(s)
    # return ddpg_agent.DDPGAgent(sess, action_space=spaces.Box(space[0], space[1], shape=space[2], dtype=np.float32),# num_actions=space,
    #                           summary_writer=summary_writer)
    return ddpg_agent_s.DDPGAgent(sess, action_space=spaces.Box(space[0], space[1], shape=space[2], dtype=np.float32),# num_actions=space,
                              summary_writer=summary_writer)
  elif FLAGS.agent_name == 'dqn':
    # return dqn_agent.DQNAgent(sess, num_actions=environment.action_space_n(),
    #                           summary_writer=summary_writer)
    os.close(s)
    return dqn_agent.DQNAgent(sess, num_actions=space[2][0],
                              summary_writer=summary_writer)
  elif FLAGS.agent_name == 'rainbow':
    # return rainbow_agent.RainbowAgent(
    #     sess, num_actions=environment.action_space_n(),
    #     summary_writer=summary_writer)
    os.close(s)
    return rainbow_agent.RainbowAgent(
        sess, num_actions=space[2][0],
        summary_writer=summary_writer)
  elif FLAGS.agent_name == 'implicit_quantile':
    # return implicit_quantile_agent.ImplicitQuantileAgent(
    #     sess, num_actions=environment.action_space_n(),
    #     summary_writer=summary_writer)
    os.close(s)
    return implicit_quantile_agent.ImplicitQuantileAgent(
        sess, num_actions=space[2][0],
        summary_writer=summary_writer)
  else:
    os.close(s)
    raise ValueError('Unknown agent: {}'.format(FLAGS.agent_name))


def create_runner(base_dir, create_agent_fn):
  """Creates an experiment Runner.

  Args:
    base_dir: str, base directory for hosting all subdirectories.
    create_agent_fn: A function that takes as args a Tensorflow session and an
     Atari 2600 Gym environment, and returns an agent.

  Returns:
    runner: A `run_experiment.Runner` like object.

  Raises:
    ValueError: When an unknown schedule is encountered.
  """
  assert base_dir is not None
  # Continuously runs training and evaluation until max num_iterations is hit.
  if FLAGS.schedule == 'continuous_train_and_eval':
    return run_experiment.Runner(base_dir, create_agent_fn)
  # Continuously runs training until max num_iterations is hit.
  elif FLAGS.schedule == 'continuous_train':
    return run_experiment.TrainRunner(base_dir, create_agent_fn)
  elif FLAGS.schedule == 'eval':
    return run_experiment.EvalRunner(base_dir, create_agent_fn)
  else:
    raise ValueError('Unknown schedule: {}'.format(FLAGS.schedule))


def launch_experiment(create_runner_fn, create_agent_fn):
  """Launches the experiment.

  Args:
    create_runner_fn: A function that takes as args a base directory and a
      function for creating an agent and returns a `Runner`-like object.
    create_agent_fn: A function that takes as args a Tensorflow session and an
     Atari 2600 Gym environment, and returns an agent.
  """
  # using default config
  run_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
  runner = create_runner_fn(FLAGS.base_dir, create_agent_fn)
  runner.run_experiment()


def main(unused_argv):
  """Main method.

  Args:
    unused_argv: Arguments (unused).
  """
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  launch_experiment(create_runner, create_agent)


if __name__ == '__main__':
  flags.mark_flag_as_required('agent_name')
  flags.mark_flag_as_required('base_dir')
  app.run(main)
