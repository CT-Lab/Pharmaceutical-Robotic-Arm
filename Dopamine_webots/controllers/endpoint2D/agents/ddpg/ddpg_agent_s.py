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
"""Compact implementation of a DDPG agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random

from replay_memory import circular_replay_buffer

import numpy as np
import tensorflow as tf
from tensorflow.initializers import random_uniform
import cv2

import gin.tf

slim = tf.contrib.slim


# NATURE_DDPG_OBSERVATION_SHAPE = (84, 84)     # for gray
NATURE_DDPG_OBSERVATION_SHAPE = (100,100,3)  # for rgb
NATURE_DDPG_DTYPE = tf.uint8
# STATE_SHAPE = (12, )
STATE_SHAPE = (8, )
STATE_DTYPE = np.float32
HIDDEN = 400
NATURE_DDPG_STACK_SIZE = 1  # Number of frames in the state stack.


def linearly_decaying_epsilon(decay_period, step, warmup_steps, epsilon):
  """Returns the current epsilon for the agent's epsilon-greedy policy.

  This follows the Nature DQN schedule of a linearly decaying epsilon (Mnih et
  al., 2015). The schedule is as follows:
    Begin at 1. until warmup_steps steps have been taken; then
    Linearly decay epsilon from 1. to epsilon in decay_period steps; and then
    Use epsilon from there on.

  Args:
    decay_period: float, the period over which epsilon is decayed.
    step: int, the number of training steps completed so far.
    warmup_steps: int, the number of steps taken before epsilon is decayed.
    epsilon: float, the final value to which to decay the epsilon parameter.

  Returns:
    A float, the current epsilon value computed according to the schedule.
  """
  steps_left = decay_period + warmup_steps - step
  bonus = (1.0 - epsilon) * steps_left / decay_period
  bonus = np.clip(bonus, 0., 1. - epsilon)
  return epsilon + bonus

class OUNoise:
  def __init__(self, mu, sigma=0.2, theta=0.15, dt=1e-2, x0=None):
    self.mu = mu
    self.theta = theta
    self.sigma = sigma
    self.dt = dt
    self.x0 = x0
    self.reset()

  def __call__(self):
    x = self.x_prev + self.theta * (self.mu-self.x_prev)*self.dt +\
      self.sigma*np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
    self.x_prev = x
    return x

  def reset(self):
    self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


@gin.configurable
class DDPGAgent(object):
  """An implementation of the DDPG."""

  def __init__(self,
               sess,
               action_space,
               num_actions=None,
               observation_shape=NATURE_DDPG_OBSERVATION_SHAPE,
               observation_dtype=NATURE_DDPG_DTYPE,
               stack_size=NATURE_DDPG_STACK_SIZE,
               state_shape=STATE_SHAPE,
               state_dtype=STATE_DTYPE,
               gamma=0.9,
               update_horizon=1,
               min_replay_history=30000, # calculate epsilon
               update_period=4,
               target_update_period=2000,
               epsilon_fn=linearly_decaying_epsilon,
               epsilon_train=0.01,
               epsilon_eval=0.001,
               epsilon_decay_period=250000,
               tf_device='/cpu:*',
               use_staging=False,
               max_tf_checkpoints_to_keep=50,
               # optimizer=tf.train.RMSPropOptimizer(
               optimizer=tf.compat.v1.train.AdamOptimizer(
                   # learning_rate=0.00025,
                   learning_rate=0.001,
                   # decay=0.95,
                   # momentum=0.0,
                   # epsilon=0.00001,
                   # centered=True),
               ),
               summary_writer=None,
               summary_writing_frequency=500):
    """Initializes the agent and constructs the components of its graph.

    Args:
      sess: `tf.Session`, for executing ops.
      num_actions: int, number of actions the agent can take at any state.
      observation_shape: tuple of ints describing the observation shape.
      observation_dtype: tf.DType, specifies the type of the observations. Note
        that if your inputs are continuous, you should set this to tf.float32.
      stack_size: int, number of frames to use in state stack.
      gamma: float, discount factor with the usual RL meaning.
      update_horizon: int, horizon at which updates are performed, the 'n' in
        n-step update.
      min_replay_history: int, number of transitions that should be experienced
        before the agent begins training its value function.
      update_period: int, period between DQN updates.
      target_update_period: int, update period for the target network.
      epsilon_fn: function expecting 4 parameters:
        (decay_period, step, warmup_steps, epsilon). This function should return
        the epsilon value used for exploration during training.
      epsilon_train: float, the value to which the agent's epsilon is eventually
        decayed during training.
      epsilon_eval: float, epsilon used when evaluating the agent.
      epsilon_decay_period: int, length of the epsilon decay schedule.
      tf_device: str, Tensorflow device on which the agent's graph is executed.
      use_staging: bool, when True use a staging area to prefetch the next
        training batch, speeding training up by about 30%.
      max_tf_checkpoints_to_keep: int, the number of TensorFlow checkpoints to
        keep.
      optimizer: `tf.train.Optimizer`, for training the value function.
      summary_writer: SummaryWriter object for outputting training statistics.
        Summary writing disabled if set to None.
      summary_writing_frequency: int, frequency with which summaries will be
        written. Lower values will result in slower training.
    """
    assert isinstance(observation_shape, tuple)
    tf.logging.info('Creating %s agent with the following parameters:',
                    self.__class__.__name__)
    tf.logging.info('\t gamma: %f', gamma)
    tf.logging.info('\t update_horizon: %f', update_horizon)
    tf.logging.info('\t min_replay_history: %d', min_replay_history)
    tf.logging.info('\t update_period: %d', update_period)
    tf.logging.info('\t target_update_period: %d', target_update_period)
    tf.logging.info('\t epsilon_train: %f', epsilon_train)
    tf.logging.info('\t epsilon_eval: %f', epsilon_eval)
    tf.logging.info('\t epsilon_decay_period: %d', epsilon_decay_period)
    tf.logging.info('\t tf_device: %s', tf_device)
    tf.logging.info('\t use_staging: %s', use_staging)
    tf.logging.info('\t optimizer: %s', optimizer)

    # self.num_actions = num_actions
    self.action_space = action_space
    self.action_low = action_space.low[0]
    self.action_high = action_space.high[0]
    self.num_actions = action_space.shape[0]
    print(action_space)
    self.noise = OUNoise(mu=np.zeros(self.num_actions))
    # self.action_dim = (num_actions, )
    self.action_dim = action_space.shape
    # self.target_name = num_actions[1]
    # self.ounoise = OUNoise(self.action_dim)    
    self.observation_shape = tuple(observation_shape)
    self.observation_dtype = observation_dtype
    self.stack_size = stack_size
    self.state_shape = tuple(state_shape)
    self.state_dtype = state_dtype
    self.gamma = gamma
    self.update_horizon = update_horizon
    self.cumulative_gamma = math.pow(gamma, update_horizon)
    self.min_replay_history = min_replay_history
    self.target_update_period = target_update_period
    self.epsilon_fn = epsilon_fn
    self.epsilon_train = epsilon_train
    self.epsilon_eval = epsilon_eval
    self.epsilon_decay_period = epsilon_decay_period
    self.update_period = update_period
    self.eval_mode = False
    self.training_steps = 0
    self.tau = 0.01
    self.optimizer = optimizer
    self.summary_writer = summary_writer
    self.summary_writing_frequency = summary_writing_frequency

    with tf.device(tf_device):
      # Create a placeholder for the state input to the DQN network.
      # The last axis indicates the number of consecutive frames stacked.

      # obs_shape = (1,) + self.observation_shape + (stack_size,) # for gray
      obs_shape = (1,) + self.observation_shape                   # for rgb

      print("obs input shape = ", obs_shape)
      self.obs = np.zeros(obs_shape)
      self.obs_ph = tf.placeholder(self.observation_dtype, obs_shape,
                                     name='obs_ph')
      s_shape = (1,) + self.state_shape
      self.state = np.zeros(s_shape)
      self.state_ph = tf.placeholder(self.state_dtype, s_shape,
                                     name = 'state_ph')
      self._replay = self._build_replay_buffer(use_staging)

      self._build_networks()

      self._train_op = self._build_train_op()
      self._sync_qt_ops, self._sync_init = self._build_sync_op()

    if self.summary_writer is not None:
      # All tf.summaries should have been defined prior to running this.
      self._merged_summaries = tf.summary.merge_all()
    self._sess = sess
    self._sess.run(tf.global_variables_initializer())
    self._saver = tf.train.Saver(max_to_keep=max_tf_checkpoints_to_keep)

    # Variables to be initialized by the agent once it interacts with the
    # environment.
    self._observation = None
    self._last_observation = None
    self._state = None
    self._last_state = None

  def _network_template_actor(self, obs):
    print("network template")
    """Builds the convolutional network used to compute the agent's Q-values.

    Args:
      state: `tf.Tensor`, contains the agent's current state.

    Returns:
      net: _network_type object containing the tensors output by the network.
    """
    # net = tf.cast(obs, tf.float32)
    # net = tf.div(net, 255.)
    # net = slim.conv2d(net, 32, [8, 8], stride=4)
    # net = slim.conv2d(net, 64, [4, 4], stride=2)
    # net = slim.conv2d(net, 64, [3, 3], stride=1)
    # net = slim.flatten(net)
    # net = slim.fully_connected(net, 300, activation_fn=tf.nn.relu)
    # net = slim.fully_connected(obs, 300, activation_fn=tf.nn.relu)
    # a = slim.fully_connected(net, self.num_actions, activation_fn=tf.nn.tanh)
    # net = tf.layers.conv2d(net, 32, [8, 8], strides=(4, 4), trainable=trainable)
    # net = tf.layers.conv2d(net, 64, [4, 4], strides=(2, 2), trainable=trainable)
    # net = tf.layers.conv2d(net, 64, [3, 3], strides=(1, 1), trainable=trainable)
    # net = tf.layers.flatten(net)

    net = tf.layers.dense(obs, HIDDEN, activation=tf.nn.relu, name='l1')
    net = tf.layers.dense(net, HIDDEN, activation=tf.nn.relu, name='l2')
    # net = tf.layers.dense(net, HIDDEN, activation=tf.nn.relu, name='l3')
    a = tf.layers.dense(net, self.num_actions, activation=tf.nn.tanh, name='a')

    # f1 = 1./np.sqrt(HIDDEN)
    # out = tf.layers.dense(obs, HIDDEN,
    #                         kernel_initializer=random_uniform(-f1, f1),
    #                         bias_initializer=random_uniform(-f1, f1), name='l1')
    # out = tf.layers.batch_normalization(out)
    # out = tf.nn.relu(out)

    # f2 = 1./np.sqrt(HIDDEN)
    # out = tf.layers.dense(out, HIDDEN,
    #                         kernel_initializer=random_uniform(-f2, f2),
    #                         bias_initializer=random_uniform(-f2, f2), name='l2')
    # out = tf.layers.batch_normalization(out)
    # out = tf.nn.relu(out)

    # f3 = 3e-3
    # out = tf.layers.dense(out, HIDDEN,
    #                         kernel_initializer=random_uniform(-f3, f3),
    #                         bias_initializer=random_uniform(-f3, f3), name='l3')
    # out = tf.layers.batch_normalization(out)
    # out = tf.nn.relu(out)

    # a = tf.layers.dense(out, self.num_actions, name='a')
    # a = tf.nn.tanh(a)


    return tf.multiply(a, self.action_high, name='scaled_a')
    # return self._get_network_type()(q_values)

  def _network_template_critic(self, state, action):
    # s_net = slim.batch_norm(state)
    # a_net = slim.batch_norm(action)
    # net = tf.concat([s_net, a_net], 1)
    # net = slim.fully_connected(net, 512, activation_fn=tf.nn.relu)
    # net = slim.batch_norm(net)
    # net = slim.fully_connected(net, 300, activation_fn=tf.nn.relu)
    # net = slim.batch_norm(net)
    # q = slim.fully_connected(net, 1, activation_fn=tf.nn.relu)

    net = tf.concat([state, action], 1)
    net = tf.layers.dense(net, HIDDEN, activation=tf.nn.relu)
    net = tf.layers.dense(net, HIDDEN, activation=tf.nn.relu)
    # net = tf.layers.dense(net, HIDDEN, activation=tf.nn.relu)
    q = tf.layers.dense(net, 1)

    # f1 = 1./np.sqrt(HIDDEN)
    # dense1 = tf.layers.dense(state, HIDDEN,
    #                         kernel_initializer=random_uniform(-f1, f1),
    #                         bias_initializer=random_uniform(-f1, f1), name='l1')
    # batch1 = tf.layers.batch_normalization(dense1)
    # layer1_activation = tf.nn.relu(batch1)

    # f2 = 1./np.sqrt(HIDDEN)
    # state_action = tf.concat([layer1_activation, action], 1)
    # dense2 = tf.layers.dense(state_action, HIDDEN,
    #                         kernel_initializer=random_uniform(-f2, f2),
    #                         bias_initializer=random_uniform(-f2, f2), name='l2')
    # # batch2 = tf.layers.batch_normalization(dense2)
    # layer2_activation = tf.nn.relu(dense2)

    # f3 = 3e-3
    # dense3 = tf.layers.dense(layer2_activation, HIDDEN,
    #                         kernel_initializer=random_uniform(-f3, f3),
    #                         bias_initializer=random_uniform(-f3, f3), name='l3')
    # batch2 = tf.layers.batch_normalization(dense3)
    # layer3_activation = tf.nn.relu(batch2)

    # q = tf.layers.dense(layer3_activation, 1)

    return q

  def _build_networks(self):
    print("build network")
    """Builds the Q-value network computations needed for acting and training.

    These are:
      self.online_convnet: For computing the current state's Q-values.
      self.target_convnet: For computing the next state's target Q-values.
      self._net_outputs: The actual Q-values.
      self._q_argmax: The action maximizing the current state's Q-values.
      self._replay_net_outputs: The replayed states' Q-values.
      self._replay_next_target_net_outputs: The replayed next states' target
        Q-values (see Mnih et al., 2015 for details).
    """
    # Calling online_convnet will generate a new graph as defined in
    # self._get_network_template using whatever input is passed, but will always
    # share the same weights.

    # ACTOR
    self.online_convnet_A = tf.make_template('Online_A', self._network_template_actor)
    self.target_convnet_A = tf.make_template('Target_A', self._network_template_actor)
    # self._net_outputs_A = self.online_convnet_A(self.obs_ph, True)
    self._net_outputs_A = self.online_convnet_A(self.state_ph)
    # self._net_outputs_A = self.online_convnet_A(self.state_ph)
    # print(self._net_outputs_A)

    # self._replay_net_outputs_A = self.online_convnet_A(self._replay.observations, True)
    # self._replay_next_target_net_outputs_A = self.target_convnet_A(self._replay.next_observations, False)
    self._replay_net_outputs_A = self.online_convnet_A(self._replay.states)
    self._replay_next_target_net_outputs_A = self.target_convnet_A(self._replay.next_states)
    # self._replay_net_outputs_A = self.online_convnet_A(self._replay.states)
    # self._replay_next_target_net_outputs_A = self.target_convnet_A(self._replay.next_states)


    # TODO(bellemare): Ties should be broken. They are unlikely to happen when
    # using a deep network, but may affect performance with a linear
    # approximation scheme.
    # self._q_argmax = tf.argmax(self._net_outputs.q_values, axis=1)[0]

    # CRITIC
    self.online_convnet_C = tf.make_template('Online_C', self._network_template_critic)
    self.target_convnet_C = tf.make_template('Target_C', self._network_template_critic)
    # self._net_outputs_C = self.online_convnet_C(self.state_ph, self._net_outputs_A)

    self._replay_net_outputs_C_for_A = self.online_convnet_C(self._replay.states, self._replay_net_outputs_A)
    # Q(s,u(o))
    self._replay_net_outputs_C_for_C = self.online_convnet_C(self._replay.states, self._replay.actions)
    self._replay_next_target_net_outputs_C = self.target_convnet_C(
      self._replay.next_states, self._replay_next_target_net_outputs_A)
    # Q(s,a)-(r+gamma(Q'(s',u(o'))))
    self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Target_A')
    self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Online_A')
    self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Target_C')
    self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Online_C')
    # print(self.at_params, self.ae_params, self.ct_params, self.ce_params)

  def _build_replay_buffer(self, use_staging):
    print("build replay buffer")
    """Creates the replay buffer used by the agent.

    Args:
      use_staging: bool, if True, uses a staging area to prefetch data for
        faster training.

    Returns:
      A WrapperReplayBuffer object.
    """
    return circular_replay_buffer.WrappedReplayBuffer(
        observation_shape=self.observation_shape,
        state_shape=self.state_shape,
        action_dim=self.action_dim,
        stack_size=self.stack_size,
        use_staging=use_staging,
        update_horizon=self.update_horizon,
        gamma=self.gamma,
        observation_dtype=self.observation_dtype.as_numpy_dtype,
        state_dtype=self.state_dtype)

  def _build_target_q_op(self):
    print("build target q op")
    """Build an op used as a target for the Q-value.

    Returns:
      target_q_op: An op calculating the Q-value.
    """
    # Get the maximum Q-value across the actions dimension.
    # replay_next_qt_max = tf.reduce_max(
    #     self._replay_next_target_net_outputs.q_values, 1)
    replay_next_q = self._replay_next_target_net_outputs_C
    # Calculate the Bellman target value.
    #   Q_t = R_t + \gamma^N * Q'_t+1
    # where,
    #   Q'_t+1 = \argmax_a Q(S_t+1, a)
    #          (or) 0 if S_t is a terminal state,
    # and
    #   N is the update horizon (by default, N=1).
    return self._replay.rewards + self.cumulative_gamma * replay_next_q * (
        1. - tf.cast(self._replay.terminals, tf.float32))

  def _build_train_op(self):
    print("build train op")
    """Builds a training op.

    Returns:
      train_op: An op performing one step of training from replay data.
    """
    # replay_action_one_hot = tf.one_hot(
    #     self._replay.actions, self.num_actions, 1., 0., name='action_one_hot')
    # replay_chosen_q = tf.reduce_sum(
    #     self._replay_net_outputs.q_values * replay_action_one_hot,
    #     reduction_indices=1,
    #     name='replay_chosen_q')

    # target = tf.stop_gradient(self._build_target_q_op())
    q_target = self._build_target_q_op()
    
    # loss = tf.losses.huber_loss(
    #     target, replay_chosen_q, reduction=tf.losses.Reduction.NONE)
    td_error = tf.losses.mean_squared_error(labels=q_target, predictions=self._replay_net_outputs_C_for_C)
    a_loss = - tf.reduce_mean(self._replay_net_outputs_C_for_A)

    if self.summary_writer is not None:
      with tf.variable_scope('C_Losses'):
        tf.summary.scalar('mean_squared_Loss', td_error)
      with tf.variable_scope('A_Losses'):
        tf.summary.scalar('reduce_mean_Loss', a_loss)
    return [self.optimizer.minimize(td_error, var_list=self.ce_params), self.optimizer.minimize(a_loss, var_list=self.ae_params)]

  def _build_sync_op(self):
    print("build sync op")
    """Builds ops for assigning weights from online to target network.

    Returns:
      ops: A list of ops assigning weights from online to target network.
    """
    # Get trainable variables from online and target DQNs
    # sync_qt_ops = []
    # trainables_online = tf.get_collection(
    #     tf.GraphKeys.TRAINABLE_VARIABLES, scope='Online')
    # trainables_target = tf.get_collection(
    #     tf.GraphKeys.TRAINABLE_VARIABLES, scope='Target')
    # for (w_online, w_target) in zip(trainables_online, trainables_target):
    #   # Assign weights from online to target network.
    #   sync_qt_ops.append(w_target.assign(w_online, use_locking=True))
    # soft replacement
    
    sync_qt_ops = [[tf.assign(ta, (1-self.tau)*ta+self.tau*ea), tf.assign(tc, (1-self.tau)*tc+self.tau*ec)]
                   for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]
    sync_qt_ops_ = [[tf.assign(ta, ea), tf.assign(tc, ec)]
                   for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

    return sync_qt_ops, sync_qt_ops_

  def begin_episode(self, observation, state):
    """Returns the agent's first action for this episode.

    Args:
      observation: numpy array, the environment's initial observation.

    Returns:
      int, the selected action.
    """
    
    self._reset_obs_state()
    self._record_obs_state(observation, state)
    

    if not self.eval_mode:
      self._train_step()
    #   self.action = self._behaviour_action()
      # self.action = self._select_action()
    self.action = self._select_action()
    # print("begin episode ", self.action)
    return self.action

  def step(self, reward, observation, state):
    """Records the most recent transition and returns the agent's next action.

    We store the observation of the last time step since we want to store it
    with the reward.

    Args:
      reward: float, the reward received from the agent's most recent action.
      observation: numpy array, the most recent observation.

    Returns:
      int, the selected action.
    """
    # print("observation ", observation)
    # print("obs shape ", observation.shape)
    self._last_observation = self._observation
    self._last_state = self._state
    self._record_obs_state(observation, state)
    # cv2.imwrite('temp.jpg', observation)
    # cv2.imwrite('r.jpg', observation[...,0])
    # cv2.imwrite('g.jpg', observation[...,1])
    # cv2.imwrite('b.jpg', observation[...,2])
    # print("_observation ", self._observation

    if not self.eval_mode:
      self._store_transition(self._last_observation, self._last_state, self.action, reward, False)
      self._train_step()
    #   self.action = self._select_action()
      
    self.action = self._select_action()
    # print("STTTTTEP ", observation.shape, state, self.action)
    # print("action ", self.action)
    return self.action

  def end_episode(self, reward):
    # print("END")
    """Signals the end of the episode to the agent.

    We store the observation of the current time step, which is the last
    observation of the episode.

    Args:
      reward: float, the last reward from the environment.
    """
    if not self.eval_mode:
      self._store_transition(self._observation, self._state, self.action, reward, True)

  # def _select_train_action(self, state):
  #   """Select an action from the set of available actions.
  
  #   Chooses an action randomly with probability self._calculate_epsilon(), and
  #   otherwise acts greedily according to the current Q-value estimates.
  
  #   Returns:
  #      int, the selected action.
  #   """
  #   epsilon = self.epsilon_eval if self.eval_mode else self.epsilon_fn(
  #       self.epsilon_decay_period,
  #       self.training_steps,
  #       self.min_replay_history,
  #       self.epsilon_train)
  # #   if random.random() <= epsilon:
  # #     # print("random")
  # #     # Choose a random action with probability epsilon.
  # #     # return random.randint(0, self.num_actions - 1)
  # #     return random.randint(0, self.total_num_action - 1)
  # #     # return 1
  # #     # return random.choices([i for i in range(self.num_actions)], k=len(self.target_name))
  # #   else:
  # #     # print("table")
  # #     # Choose the action with highest Q-value at the current state.
  # #     return int(self._sess.run(self._q_argmax, {self.state_ph: self.state}))
  #   if epsilon > 1:
  #       # print(epsilon)
  #       # action = self._sess.run(self._net_outputs_A, {self.obs_ph: self.obs})[0]
  #       action = self._sess.run(self._net_outputs_A, {self.state_ph: self.state})[0]
  #       action = np.random.normal(action, epsilon/15)
  #   else:
  #     # print(epsilon)
  #     action = self.bc_model.predict(state[:5], state[5:12], [0,1.57])
  #     # action = np.array([0.001]*7)
  #   # action = self._sess.run(self._net_outputs_A, {self.obs_ph: self.obs})[0]
  #   action = action.clip(self.action_low, self.action_high)
    
    # return action
  # def _select_action(self):
  #   epsilon = self.epsilon_eval if self.eval_mode else self.epsilon_fn(
  #       self.epsilon_decay_period,
  #       self.training_steps,
  #       self.min_replay_history,
  #       self.epsilon_train)
  #   if random.random() <= epsilon:
  #     # Choose a random action with probability epsilon.
  #     return np.random.uniform(-0.016, 0.016, self.num_actions)
  #   else:
  #     return self._sess.run(self._net_outputs_A, {self.state_ph: self.state})[0]


  def _select_action(self):
    a = self._sess.run(self._net_outputs_A, {self.state_ph: self.state})[0]
    if self.eval_mode:
      return np.clip(a, -1, 1)
    else:
      # epsilon = self.epsilon_eval if self.eval_mode else self.epsilon_fn(
      #           self.epsilon_decay_period,
      #           self.training_steps,
      #           self.min_replay_history,
      #           self.epsilon_train)
      return np.clip(a + self.noise(), -1, 1)

    # return self._sess.run(self._net_outputs_A, {self.obs_ph: self.obs})[0]
    # return self._sess.run(self._net_outputs_A, {self.state_ph: self.state})[0]

  def _train_step(self):
    # print("train step")
    """Runs a single training step.

    Runs a training op if both:
      (1) A minimum number of frames have been added to the replay buffer.
      (2) `training_steps` is a multiple of `update_period`.

    Also, syncs weights from online to target network if training steps is a
    multiple of target update period.
    """
    # Run a train op at the rate of self.update_period if enough training steps
    # have been run. This matches the Nature DQN behaviour.
    if self._replay.memory.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        # print('run train')
        self._sess.run(self._train_op)
        if (self.summary_writer is not None and
            self.training_steps > 0 and
            self.training_steps % self.summary_writing_frequency == 0):
          summary = self._sess.run(self._merged_summaries)
          self.summary_writer.add_summary(summary, self.training_steps)

      if self.training_steps % self.target_update_period == 0:
        # print('run syncccccccccccccccccccccccccccccccccccccccccccccccccccccccc')
        self._sess.run(self._sync_qt_ops)

    self.training_steps += 1

  def _record_obs_state(self, observation, state):
    # print("record obs", state)
    """Records an observation and update state.

    Extracts a frame from the observation vector and overwrites the oldest
    frame in the state buffer.

    Args:
      observation: numpy array, an observation from the environment.
    """
    # Set current observation. We do the reshaping to handle environments
    # without frame stacking.
    observation = np.reshape(observation, self.observation_shape)
    self._observation = observation[..., 0]
    self._observation = np.reshape(observation, self.observation_shape)
    # Swap out the oldest frame with the current frame.
    self.obs = np.roll(self.obs, -1, axis=-1)
    # self.obs[0, ..., -1] = self._observation # for gray
    self.obs[0,...] = self._observation        # for rgb
    self._state = state
    self.state[0,...] = state

  def _store_transition(self, last_observation, last_state, action, reward, is_terminal):
    # print("store transition")
    """Stores an experienced transition.

    Executes a tf session and executes replay buffer ops in order to store the
    following tuple in the replay buffer:
      (last_observation, action, reward, is_terminal).

    Pedantically speaking, this does not actually store an entire transition
    since the next state is recorded on the following time step.

    Args:
      last_observation: numpy array, last observation.
      action: int, the action taken.
      reward: float, the reward.
      is_terminal: bool, indicating if the current state is a terminal state.
    """
    self._replay.add(last_observation, last_state, action, reward, is_terminal)

  def _reset_obs_state(self):
    # print("reset")
    """Resets the agent state by filling it with zeros."""
    self.obs.fill(0)
    self.state.fill(0)

  def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
    """Returns a self-contained bundle of the agent's state.

    This is used for checkpointing. It will return a dictionary containing all
    non-TensorFlow objects (to be saved into a file by the caller), and it saves
    all TensorFlow objects into a checkpoint file.

    Args:
      checkpoint_dir: str, directory where TensorFlow objects will be saved.
      iteration_number: int, iteration number to use for naming the checkpoint
        file.

    Returns:
      A dict containing additional Python objects to be checkpointed by the
        experiment. If the checkpoint directory does not exist, returns None.
    """
    if not tf.gfile.Exists(checkpoint_dir):
      return None
    # Call the Tensorflow saver to checkpoint the graph.
    self._saver.save(
        self._sess,
        os.path.join(checkpoint_dir, 'tf_ckpt'),
        global_step=iteration_number)
    # Checkpoint the out-of-graph replay buffer.
    self._replay.save(checkpoint_dir, iteration_number)
    bundle_dictionary = {}
    bundle_dictionary['observation'] = self.obs
    bundle_dictionary['state'] = self.state
    bundle_dictionary['eval_mode'] = self.eval_mode
    bundle_dictionary['training_steps'] = self.training_steps
    return bundle_dictionary

  def unbundle(self, checkpoint_dir, iteration_number, bundle_dictionary):
    """Restores the agent from a checkpoint.

    Restores the agent's Python objects to those specified in bundle_dictionary,
    and restores the TensorFlow objects to those specified in the
    checkpoint_dir. If the checkpoint_dir does not exist, will not reset the
      agent's state.

    Args:
      checkpoint_dir: str, path to the checkpoint saved by tf.Save.
      iteration_number: int, checkpoint version, used when restoring replay
        buffer.
      bundle_dictionary: dict, containing additional Python objects owned by
        the agent.

    Returns:
      bool, True if unbundling was successful.
    """
    try:
      # self._replay.load() will throw a NotFoundError if it does not find all
      # the necessary files, in which case we abort the process & return False.
      self._replay.load(checkpoint_dir, iteration_number)
    except tf.errors.NotFoundError:
      return False
    for key in self.__dict__:
      if key in bundle_dictionary:
        self.__dict__[key] = bundle_dictionary[key]
    # Restore the agent's TensorFlow graph.
    self._saver.restore(self._sess,
                        os.path.join(checkpoint_dir,
                                     'tf_ckpt-{}'.format(iteration_number)))
    return True
