import os
import time
import json
import numpy as np
import random
# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, LED, DistanceSensor
from controller import Robot
from controller import Supervisor
from controller import Motor
from controller import Camera
from controller import DistanceSensor
from controller import TouchSensor
# import robot_env
import endpoint_env
import preprocessing
from pipe import make_pipe, close_pipe, open_write_pipe, open_read_pipe, write_to_pipe, read_from_pipe

# create the Robot instance.
# env = endpoint_env2D_simple.EndP_env2D()
env = endpoint_env.EndP_env()
# env = endpoint_env.EndP_env3D_old()
# env = endpoint_env.EndP_env3D1()
# env = robot_env.Robot_env()
env = preprocessing.RobotPreprocessing(env)
# rl = DDPG(2, 9, [-1, 1])

# get the time step of the current world.
timestep = env.timestep

# Environment Reset
def reset():
  env.reset()

def unreload_reset():
  env.unreload_reset()
# Environment Step
def step(action_index):
  obs, state, reward, is_terminal, info = env.step(action_index)
  # print(obs.shape, state, reward, is_terminal)
  return obs.tolist(), state, reward, is_terminal, info
  # return {"observation":obs.tolist(), "reward":reward,
  #         "is_terminal":is_terminal}

# Environment Initial -> Go timestep & get observation
def initial_step():
  return env.after_reset()
# For Creating Agent Parameter
def action_space_info():
  info = []
  action_space = env.action_space
  info.append(float(action_space.low[0]))
  info.append(float(action_space.high[0]))
  info.append(action_space.shape)
  return info

def action_num():
  return env.action_num

def motor_name():
  return env.motor_name

def done():
  return env.done


# share pipe in every slave
channel_name = "/tmp/channel_in.pipe"
space_name = "/tmp/space_out.pipe"

# pipe head name
action_name = "/tmp/action_in"
obs_name = "/tmp/obs_out"
touch_name = "/tmp/touch_out"
reward_name = "/tmp/reward_out"
over_name = "/tmp/over_out"
terminal_name = "/tmp/term_out"
reset_name = "/tmp/reset_in"
read_name = [action_name, reset_name]
write_name = [obs_name, touch_name, reward_name, over_name, terminal_name]

make_pipe(channel_name)
make_pipe(space_name)
make_pipe('/tmp/complete.pipe')


space_pipe = open_write_pipe(space_name)
channel_pipe = open_read_pipe(channel_name)
channel = read_from_pipe(channel_pipe, 1)

print(action_num())
if channel == 0:
  complete_pipe = open_read_pipe("/tmp/complete.pipe")
  complete = read_from_pipe(complete_pipe, 1)
  if not complete:
    print("write space")
    write_to_pipe(space_pipe, action_space_info())
  # os.close(complete_pipe)
  close_pipe(complete_pipe)
print("I AM CHANNEL %s"%channel)


# head + tail name pipe
read_name_list = [(i + "%s.pipe"%channel) for i in read_name]
write_name_list = [(i + "%s.pipe"%channel) for i in write_name]
all_path = read_name_list + write_name_list
print(all_path)
make_pipe(all_path)

obs_pipe, touch_pipe, reward_pipe, over_pipe, terminal_pipe = open_write_pipe(write_name_list)
action_pipe, reset_pipe = open_read_pipe(read_name_list)
'''
initial_obs, initial_state = initial_step()
write_to_pipe([obs_pipe, touch_pipe], [initial_obs, initial_state])
print(np.array(initial_obs).shape, initial_state)
'''
initial_state = initial_step()
write_to_pipe(touch_pipe, initial_state)
# np.set_printoptions(8)
# while env.robot.step(timestep) != -1:
while True:

  action = read_from_pipe(action_pipe, 1024)
  # distribute action type
  obs, state, reward, is_terminal, info = step(action)
  # print(state)
  # print(action)
  # print(np.array(state[-7:])*2*np.pi)
  '''write_to_pipe([touch_pipe, reward_pipe, terminal_pipe, over_pipe, obs_pipe], [state, reward, is_terminal, done(), obs])'''
  write_to_pipe([touch_pipe, reward_pipe, terminal_pipe, over_pipe], [state, reward, is_terminal, done()])
  rst = read_from_pipe(reset_pipe, 20)
  
  if rst:
    close_pipe([action_pipe,space_pipe,obs_pipe,touch_pipe,reward_pipe,over_pipe,terminal_pipe,reset_pipe,channel_pipe])
    reset()

# Enter here exit cleanup code.
