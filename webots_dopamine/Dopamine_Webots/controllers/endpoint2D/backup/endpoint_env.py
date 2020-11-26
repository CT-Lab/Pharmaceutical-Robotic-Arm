from controller import Robot
from controller import Supervisor
from controller import Motor
from controller import PositionSensor
from controller import DistanceSensor
from controller import Camera
from controller import Node
from controller import Field

from gym import spaces
import os, random
import numpy as np
# import pandas as pd
import time
import pickle
import math
from time import gmtime, strftime, sleep

# import cv2

class EndP_env3D(object):
  """docstring for Robot_arm"""
  def __init__(self):

    # ---create robot as supervisor---
    self.robot = Supervisor()
    
    # ---get simulation timestep---
    self.timestep = int(self.robot.getBasicTimeStep())
    '''Position Motors'''
    self.velocity = 1
    self.dt = 0.032
    
    ## to be modified : init_position
    '''2D 2Motor'''
    # self.rm1_init_position = 0
    # self.rm4_init_position = -0.0698
    # '''3D 4Motor train'''
    # self.rm2_init_position = 0
    # self.rm3_init_position = 0
    # self.rm5_init_position = 0
    # self.rm6_init_position = 0
    # self.rm7_init_position = 0
    self.rm1_init_position = self.robot.getFromDef("Franka_Emika_Panda.joint01.j1p").getField('position').getSFFloat()
    print("rm1_initpos: {}".format(self.rm1_init_position))
    self.rm2_init_position = self.robot.getFromDef("Franka_Emika_Panda.joint01.link1.joint12.j2p").getField('position').getSFFloat()
    self.rm3_init_position = self.robot.getFromDef("Franka_Emika_Panda.joint01.link1.joint12.link2.joint23.j3p").getField('position').getSFFloat()
    self.rm4_init_position = self.robot.getFromDef("Franka_Emika_Panda.joint01.link1.joint12.link2.joint23.link3.joint34.j4p").getField('position').getSFFloat()
    self.rm5_init_position = self.robot.getFromDef("Franka_Emika_Panda.joint01.link1.joint12.link2.joint23.link3.joint34.link4.joint45.j5p").getField('position').getSFFloat()
    self.rm6_init_position = self.robot.getFromDef("Franka_Emika_Panda.joint01.link1.joint12.link2.joint23.link3.joint34.link4.joint45.link5.joint56.j6p").getField('position').getSFFloat()
    self.rm7_init_position = self.robot.getFromDef("Franka_Emika_Panda.joint01.link1.joint12.link2.joint23.link3.joint34.link4.joint45.link5.joint56.link6.joint67.j7p").getField('position').getSFFloat()
    '''Orientation Motor'''

    ## for franka
    self.P2 = self.robot.getFromDef("Franka_Emika_Panda.joint01.link1.joint12.link2.M2P")
    self.P4 = self.robot.getFromDef("Franka_Emika_Panda.joint01.link1.joint12.link2.joint23.link3.joint34.link4.M4P")
    self.ENDP = self.robot.getFromDef("Franka_Emika_Panda.joint01.link1.joint12.link2.joint23.link3.joint34.link4.joint45.link5.joint56.link6.joint67.link7.joint7H.hand.endEffector.endp")

    self.origin = np.array(self.P2.getPosition())
    # self.elbowP = np.array(self.P4.getPosition())
    self.endpointP = np.array(self.ENDP.getPosition())
    self.edge = self.endpointP[0]-self.origin[0]
    print("edge : {}".format(self.edge))
    print("NEW")

    self.beaker = self.robot.getFromDef('Beaker')
    print("beaker_ori: {}".format(self.beaker.getOrientation()[4]))

    ## to be modified : modify the random space to fit in our robot's work place
    ## we should use 3D random goal

    ## for franka
    ## use a box space for now
    #x_range = np.linspace(0.4, 0.7, 2)
    #x = np.random.choice(x_range)
    #z_range = np.linspace(-0.25, 0.25, 3)
    #z = np.random.choice(z_range)
    x = 0.4
    z = 0.1
    # y_range = np.linspace(0.0, 0.3, 3)
    # y = np.random.choice(y_range)
    y = 0.15
    self.goal_position = self.robot.getFromDef('TARGET.tar').getPosition()
    print("goal : {}".format(self.goal_position))
    '''3D goal'''
    self.goal_np_p = np.array(self.goal_position)
    d, _ = self._get_elbow_position()
    print("(elbow_local/self.edge) : {}".format(d))
    # self.GOAL = self.robot.getFromDef('goal')
    # self.GOAL_P = self.GOAL.getField('translation')
    # self.GOAL_P.setSFVec3f(self.goal_position)

    ## for franka
    self.arm_motor_name = [
      # 'motor1', 
      'motor2', 'motor3', 'motor4', 'motor5', 'motor6']
    '''3D 4Motor'''
    self.arm_position = dict(zip(self.arm_motor_name, [
      # self.rm1_init_position, 
      self.rm2_init_position, self.rm3_init_position, self.rm4_init_position, self.rm5_init_position, self.rm6_init_position]))
                                                      #  self.rm7_init_position]))
    self.arm_motors, self.arm_ps = self.create_motors(self.arm_motor_name, self.arm_position)
    
    ## to be modified : set the position range
    self.arm_position_range = np.array([
      # [-2.897, 2.897], 
    [-1.763, 1.763], [-2.8973, 2.8973],
                                        [-3.072, -0.0698], [-2.8973, 2.8973], [-0.0175, 3.7525]])
                                        # [-2.897, 2.897]])
    self.act_mid = dict(zip(self.arm_motor_name, np.mean(self.arm_position_range, axis=1)))
    self.act_rng = dict(zip(self.arm_motor_name, 0.5*(self.arm_position_range[:, 1] - self.arm_position_range[:, 0])))
    self.arm_position_range = dict(zip(self.arm_motor_name, [
      # [-2.897, 2.897], 
      [-1.763, 1.763], [-2.8973, 2.8973],
                                                            [-3.072, -0.0698], [-2.8973, 2.8973], [-0.0175, 3.7525]]))
                                                            # [-2.897, 2.897]]))
    '''Orientation Motors'''   

    # ---create camera dict---
    self.cameras = {}
    self.camera_name = ['vr_camera_r',
                        # "camera_center"
    ]
    self.cameras = self.create_sensors(self.camera_name, self.robot.getCamera)
    
    # ---get image height and width---
    self.camera_height = self.cameras["vr_camera_r"].getHeight()
    self.camera_width = self.cameras["vr_camera_r"].getWidth()
    # ---set game over for game start---
    self.done = False
    self.lives = 0
    # ---action set for motor to rotate clockwise/counterclockwise or stop
    
    self.action_num = len(self.arm_motors)
    print("action_num: {}".format(self.action_num))
    self.action_space = spaces.Box(-1, 1, shape=(self.action_num,), dtype=np.float32)
    # self.action_space = spaces.Box(-0.016, 0.016, shape=(self.action_num, ), dtype=np.float32)
    
    self.on_goal = 0
    # self.crash = 0

    self.accumulated_reward = 0
    self.episode_len_reward = 0
    self.episode_ori_reward = 0
    self.arrival_time = 0
    # print(self.ENDP.getOrientation())
    self.robot.step(self.timestep)
    self.start_time = self.getTime()

    '''constant'''
    self.NEAR_DISTANCE = 0.05
    self.BONUS_REWARD = 10
    self.ON_GOAL_FINISH_COUNT = 5
    self.MAX_STEP = 100

    '''variable'''
    self.finished_step = 0
    self.prev_d = self.goal_np_p - np.array(self.ENDP.getPosition())
    self.left_bonus_count = self.ON_GOAL_FINISH_COUNT

  def reset_eval_to_down(self):
    self.on_goal = 0
    # self.crash = 0

    self.accumulated_reward = 0
    self.episode_len_reward = 0
    self.episode_ori_reward = 0
    self.arrival_time = 0
    self.finished_step = 0
    self.prev_d = self.goal_np_p - np.array(self.ENDP.getPosition())
    self.left_bonus_count = self.ON_GOAL_FINISH_COUNT
    self.start_time = self.getTime()
    self.done = False
    self.lives = 0

  def _get_robot(self):
    return self.robot

  def create_motors(self, names, position):
    obj = {}
    ps_obj = {}
    for name in names:
      motor = self.robot.getMotor(name)
      motor.setVelocity(self.velocity)
      motor.setPosition(position[name])
      ps = motor.getPositionSensor()
      ps.enable(self.timestep)
      obj[name] = motor
      ps_obj[name] = ps
    return obj, ps_obj

  def create_sensors(self, names, instant_func):
    obj = {}
    for name in names:
      sensor = instant_func(name)
      sensor.enable(self.timestep)
      obj[name] = sensor
    return obj

  def set_goal_position(self, arg):
    #x_range = np.linspace(0.4, 0.7, 2)
    #x = np.random.choice(x_range)
    #z_range = np.linspace(-0.25, 0.25, 3)
    #z = np.random.choice(z_range)
    # y_range = np.linspace(0.0, 0.3, 3)
    # y = np.random.choice(y_range)
    x = 0.5665
    z = 0.0241
    y = 0.66
    # self.goal_position = [x, y, z]
    self.goal_position = self.robot.getFromDef('TARGET.tar').getPosition()
    if arg=='down':
      self.goal_position = [x, y, z]
      self.robot.getFromDef('TARGET').getField('translation').setSFVec3f(self.goal_position)

    self.goal_np_p = np.array(self.goal_position)
    # self.GOAL_P.setSFVec3f(self.goal_position)
    self.robot.step(self.timestep)

  def _get_image(self):
    return self.cameras["vr_camera_r"].getImage()

  def _get_obs(self):
    return self._get_image()

  def _get_motor_position(self):
    motor_position = []
    for name in self.arm_motor_name:
      data = self.arm_ps[name].getValue()
      # normalized
      data = (data - self.act_mid[name])/self.act_rng[name]
      motor_position.append(data)
    return motor_position
  
  def _get_endpoint_orientation(self):
    orientation = np.array(self.ENDP.getOrientation())
    # print(orientation[::4])
    d = self.on_goal - orientation[::4]
    return orientation.tolist(), (d/2).tolist()

  def _get_state(self):
    state = []
    '''Position'''
    elp, elbd = self._get_elbow_position()
    arm_p = self._get_motor_position()
    ep, ed = self._get_endpoint_position()

    ## to be modified : choose what to be our state
    # state += elp+ep+elbd+ed+[1. if self.on_goal else 0.]
    state += arm_p + ed
    '''Orientation'''
    # orient, d = self._get_endpoint_orientation()
    # state += orient+d+[1. if self.on_goal else 0.]
    
    return state
    
  def _get_elbow_position(self):
    
    goal = self.goal_np_p
    '''3D Position Distance'''
    ## to be modified : set elbow; why edge?
    elbow_global = np.array(self.P4.getPosition())
    elbow_local = elbow_global - self.origin
    d = (goal - elbow_global)
    return (elbow_local/self.edge).tolist(), (d/self.edge).tolist()

  def _get_endpoint_position(self):
    goal = self.goal_np_p
    '''3D Position Distance'''
    end_global = np.array(self.ENDP.getPosition())
    end_local = end_global - self.origin
   
    d = goal - end_global
    return (end_local/self.edge).tolist(), (d/self.edge).tolist()
  
  ## to be modified
  def _get_reward(self):
    '''Position Reward'''
    _, d = self._get_endpoint_position()
    d = np.array(d)*self.edge
    # d = np.array(d)*0.565*2
    
    '''3D reward'''
    # r = -np.sqrt(d[0]**2+d[1]**2+d[2]**2)
    r = np.linalg.norm(self.prev_d) - np.linalg.norm(d)
    distance = np.linalg.norm(d)

    if distance < self.NEAR_DISTANCE :
      if self.left_bonus_count > 0 :
        bonus = self.BONUS_REWARD * (self.MAX_STEP - 2*self.finished_step) / self.MAX_STEP
        r += bonus
        self.left_bonus_count -= 1
        print('bonus: ', bonus)
      
      self.on_goal += 1
      print('on goal: ', self.on_goal)
      if self.on_goal >= self.ON_GOAL_FINISH_COUNT :
        self.done = True
        print('ON GOAL')
    else:
      self.on_goal = 0
      print('on goal: 0')
    
    self.episode_len_reward += r

    self.orientation_reward = 1000*(self.beaker.getOrientation()[4] - 0.001)
    # print("ori: {}".format(self.beaker.getOrientation()[4]))
    print("ori_reward: {}".format(self.orientation_reward))
    r += self.orientation_reward

    self.episode_ori_reward += self.orientation_reward
    self.accumulated_reward += r
    self.prev_d = d
    self.finished_step += 1

    return r
  
  @property
  def obsrvation_space(self):
    return spaces.Box(low=0, high=255, shape=(self.camera_height, self.camera_width, 3), dtype=np.uint8) # for rgb

  def action_space(self):
    return spaces.Box(-1, 1, shape=(self.action_num, ), dtype=np.float32)

  def get_lives(self):
    return self.lives

  def getTime(self):
    return self.robot.getTime()
    
  def reset(self):
    
    fp = open("Episode-len-score.txt","a")
    fp.write(str(self.episode_len_reward)+'\n')
    fp.close()
    
    fp = open("Episode-orientation-score.txt","a")
    fp.write(str(self.episode_ori_reward)+'\n')
    fp.close()
    
    fp = open("Episode-score.txt","a")
    fp.write(str(self.accumulated_reward)+'\n')
    fp.close()

    self.robot.worldReload()

  def step(self, action_dict):
    reward = 0
    action_dict = np.clip(action_dict, -1, 1)
    for name, a in zip(self.arm_motor_name, action_dict):
      self.arm_position[name] += a * self.dt
      if self.arm_position_range[name][0] > self.arm_position[name]:
        self.arm_position[name] = self.arm_position_range[name][0]
      elif self.arm_position_range[name][1] < self.arm_position[name]:
        self.arm_position[name] = self.arm_position_range[name][1]
      self.arm_motors[name].setPosition(self.arm_position[name])
    self.robot.step(self.timestep)

    reward = self._get_reward()
    
    state = self._get_state()
    obs = False

    return obs, state, reward, self.done, {"goal_achieved":self.on_goal>0}
