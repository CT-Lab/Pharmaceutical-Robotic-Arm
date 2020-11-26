from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from controller import Robot
from controller import Motor
from controller import DistanceSensor
from controller import Camera

from gym.spaces.box import Box
import numpy as np
import cv2

class RobotPreprocessing(object):
  """docstring for Robot_arm"""
  def __init__(self, environment, frame_skip=1, terminal_on_life_lose=False,
               screen_size=100):
    
    if frame_skip <= 0:
      raise ValueError('Frame skip should be strictly positive, got {}'.
                       format(frame_skip))
    if screen_size <= 0:
      raise ValueError('Target screen size should be strictly positive, got {}'.
                       format(screen_size))

    self.environment = environment
    self.terminal_on_life_lose = terminal_on_life_lose
    self.frame_skip = frame_skip
    self.screen_size = screen_size

    
    obs_dims = self.environment.obsrvation_space
    self.image_buffer = np.empty((obs_dims.shape[0], obs_dims.shape[1], 3), dtype=np.uint8),

    self.screen_buffer = [
        np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8),
        np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8)
    ]
    self.screen_rgb_buffer = [
        np.empty((obs_dims.shape[0], obs_dims.shape[1], 3), dtype=np.uint8),
        np.empty((obs_dims.shape[0], obs_dims.shape[1], 3), dtype=np.uint8)
    ]

    self.done = False
    self.lives = 1
    # print("WEBOTSPREPROCESSING")
    # print("env from webots ", self.environment)
    # print("terminal_on_life_lose ", self.terminal_on_life_lose)
    # print("frame_skip ", self.frame_skip)
    # print("screen size ", self.screen_size)
    # print("obs_dims ", obs_dims)
    # print("screen_buffer size ", len(self.screen_buffer))
    # print("done ", self.done)
    # print("lives ", self.lives)
    # print("action_space", self.action_space)
  
  @property
  def obsrvation_space(self):
    return Box(low=0, high=255, shape=(self.screen_size, self.screen_size, 3), dtype=np.uint8) # for rgb
    # return Box(low=0, high=255, shape=(self.screen_size, self.screen_size), dtype=np.uint8) #for grey

  @property
  def action_space(self):
    return self.environment.action_space

  @property
  def action_num(self):
    return self.environment.action_num

  @property
  def motor_name(self):
    return self.environment.arm_motor_name

  @property
  def reward_range(self):
    return self.environment.get_reward_range

  def metadata(self):
    pass

  @property
  def timestep(self):
    return self.environment.timestep

  @property
  def robot(self):
    return self.environment.robot

  def reset(self):
    self.environment.reset()

  def unreload_reset(self):
    self.environment.unreload_reset()
    
  def after_reset(self):
    self.robot.step(self.timestep)
    '''
    self._fetch_grayscale_observation(self.screen_buffer[1])
    self._fetch_rgb_observation(self.screen_rgb_buffer[1])
    self.screen_buffer[0].fill(0)
    self.screen_rgb_buffer[0].fill(0)
    '''
    init_state = self.environment._get_state()
    '''return self._pool_and_resize(rgb=True).tolist(), init_state'''
    return init_state


  # webots no need to use rander
  def render(self, mode):
    pass 

  def step(self, action):
    accumulated_reward = 0
    
    for time_step in range(self.frame_skip):

      _, state, reward, done, info = self.environment.step(action)
      accumulated_reward += reward

      if self.terminal_on_life_lose:
        new_lives = self.environment.get_lives()
        is_termial = done or new_lives < self.lives
        self.lives = new_lives
      else:
        is_termial = done

      if is_termial:
        break
      elif time_step >= self.frame_skip - 2:
        t = time_step - (self.frame_skip - 2)
        '''
        self._fetch_grayscale_observation(self.screen_buffer[t])
        self._fetch_rgb_observation(self.screen_rgb_buffer[t])

    observation = self._pool_and_resize(rgb=True)
    '''
    observation = np.zeros([100,100,3])
    # cv2.imwrite('fs1.jpg', observation)

    self.done = done
    return observation, state, accumulated_reward, is_termial, info

  def _fetch_rgb_observation(self, output):
    img = self.environment._get_obs()
    # self.environment.cameras['camera'].saveImage('./tmp.jpg', 100)
    for x in range(self.environment.camera_width):
      for y in range(self.environment.camera_height):
        output[y][x][2] = self.environment.cameras['camera'].imageGetRed(img, self.environment.camera_width, x, y)
        output[y][x][1] = self.environment.cameras['camera'].imageGetGreen(img, self.environment.camera_width, x, y)
        output[y][x][0] = self.environment.cameras['camera'].imageGetBlue(img, self.environment.camera_width, x, y)
    # ot = cv2.imread('tmp.jpg')
    # cv2.imwrite('tmp2.jpg', ot)
    # print('output', type(output))
    return output


  def _fetch_grayscale_observation(self, output):
    img = self.environment._get_obs()
    # self.environment.cameras['camera'].saveImage('./tmp.jpg', 100)
    for x in range(self.environment.camera_width):
      for y in range(self.environment.camera_height):
        output[y][x] = self.environment.cameras['camera'].imageGetGray(img, self.environment.camera_width, x, y)
    return output

  def _pool_and_resize(self, rgb):
    """Transforms two frames into a Nature DQN observation.

      For efficiency, the transformation is done in-place in self.screen_buffer.

    Returns:
      transformed_screen: numpy array, pooled, resized screen.
    """
    # Pool if there are enough screens to do so.
    if rgb:
      if self.frame_skip > 1:
        np.maximum(self.screen_rgb_buffer[0], self.screen_rgb_buffer[1],
                   out=self.screen_rgb_buffer[1])

      transformed_image = cv2.resize(self.screen_rgb_buffer[1],
                                     (self.screen_size, self.screen_size),
                                     interpolation=cv2.INTER_AREA)
      int_image = np.asarray(transformed_image, dtype=np.uint8)
      # return np.expand_dims(int_image, axis=3)
      return int_image
    else:
      if self.frame_skip > 1:
        np.maximum(self.screen_buffer[0], self.screen_buffer[1],
                   out=self.screen_buffer[1])

      transformed_image = cv2.resize(self.screen_buffer[1],
                                     (self.screen_size, self.screen_size),
                                     interpolation=cv2.INTER_AREA)
      int_image = np.asarray(transformed_image, dtype=np.uint8)
      return np.expand_dims(int_image, axis=2)
