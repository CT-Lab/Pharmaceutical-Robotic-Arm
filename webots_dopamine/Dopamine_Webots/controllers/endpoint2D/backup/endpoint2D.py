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
import endpoint_env
import preprocessing
from pipe import make_pipe, close_pipe, open_write_pipe, open_read_pipe, write_to_pipe, read_from_pipe

from enum import Enum

# create the Robot instance.
env = endpoint_env.EndP_env3D()

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

def output(str):
  print(str)

# share pipe in every slave
channel_name = "/tmp/channel_in1.pipe"
space_name = "/tmp/space_out1.pipe"
goal_name = "/tmp/goal_in1.pipe"

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
output("ready to make pipe")
make_pipe(channel_name)
make_pipe(space_name)
make_pipe(goal_name)
make_pipe('/tmp/complete.pipe')
output("make pipe channel, space, goal")

space_pipe = open_write_pipe(space_name)
output("open read pipe space")
channel_pipe = open_read_pipe(channel_name)
output("open read pipe channel")
channel = read_from_pipe(channel_pipe, 1)
output("read from pipe channel: {}".format(channel))

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

'''evaluation use'''
goal_pipe = open_read_pipe(goal_name)
goal_info = read_from_pipe(goal_pipe)
close_pipe(goal_pipe)

env.environment.set_goal_position(goal_info)
'''.................................'''

# head + tail name pipe
read_name_list = [(i + "%s.pipe"%(channel+1)) for i in read_name]
write_name_list = [(i + "%s.pipe"%(channel+1)) for i in write_name]
all_path = read_name_list + write_name_list
# print(all_path)
make_pipe(all_path)
obs_pipe, touch_pipe, reward_pipe, over_pipe, terminal_pipe = open_write_pipe(write_name_list)
action_pipe, reset_pipe = open_read_pipe(read_name_list)

# FOR DOWN
# share pipe in every slave
channel_name_down = "/tmp/channel_in1_down.pipe"
space_name_down = "/tmp/space_out1_down.pipe"
goal_name_down = "/tmp/goal_in1_down.pipe"

# pipe head name
action_name_down = "/tmp/action_in_down"
obs_name_down = "/tmp/obs_out_down"
touch_name_down = "/tmp/touch_out_down"
reward_name_down = "/tmp/reward_out_down"
over_name_down = "/tmp/over_out_down"
terminal_name_down = "/tmp/term_out_down"
reset_name_down = "/tmp/reset_in_down"
read_name_down = [action_name_down, reset_name_down]
write_name_down = [obs_name_down, touch_name_down, reward_name_down, over_name_down, terminal_name_down]
output("ready to make pipe_down")
make_pipe(channel_name_down)
make_pipe(space_name_down)
make_pipe(goal_name_down)
make_pipe('/tmp/complete_down.pipe')
output("make pipe_down channel, space, goal")

space_pipe_down = open_write_pipe(space_name_down)
output("open read pipe space_down")
channel_pipe_down = open_read_pipe(channel_name_down)
output("open read pipe channel_down")
channel_down = read_from_pipe(channel_pipe_down, 1)
output("read from pipe channel_down: {}".format(channel_down))

print(action_num())
if channel_down == 0:
  complete_pipe_down = open_read_pipe("/tmp/complete_down.pipe")
  complete_down = read_from_pipe(complete_pipe_down, 1)
  if not complete_down:
    print("write space")
    write_to_pipe(space_pipe_down, action_space_info())
  # os.close(complete_pipe)
  close_pipe(complete_pipe_down)
print("I AM CHANNEL %s"%channel_down)

'''evaluation use'''
goal_pipe_down = open_read_pipe(goal_name_down)
goal_info = read_from_pipe(goal_pipe_down)
close_pipe(goal_pipe_down)

# env.environment.set_goal_position(goal_info)
'''.................................'''

# head + tail name pipe
read_name_list = [(i + "%s.pipe"%(channel_down+1)) for i in read_name_down]
write_name_list = [(i + "%s.pipe"%(channel_down+1)) for i in write_name_down]
all_path = read_name_list + write_name_list
# print(all_path)
make_pipe(all_path)

obs_pipe_down, touch_pipe_down, reward_pipe_down, over_pipe_down, terminal_pipe_down = open_write_pipe(write_name_list)
action_pipe_down, reset_pipe_down = open_read_pipe(read_name_list)

CAM_A = Camera("CAM_A")
CAM_A.enable(32)
CAM_A.recognitionEnable(32)

'''
initial_obs, initial_state = initial_step()
write_to_pipe([obs_pipe, touch_pipe], [initial_obs, initial_state])
print(np.array(initial_obs).shape, initial_state)
'''
initial_state = initial_step()
print("init_state: {}".format(initial_state))
write_to_pipe(touch_pipe, initial_state)

class TaskType(Enum):
  UP = 'up_dopamine'
  DOWN = 'down_dopamine'
  GRAB = 'grab_ik'
  POUR = 'pour_ik'
  RELEASE = 'release_ik'

taskList = [
  TaskType.UP,
  TaskType.DOWN
]
task_done = True

# while env.robot.step(timestep) != -1:
while True:

  if taskList or not task_done:
    if task_done:
      current_task = taskList.pop(0)
      task_done = False
  else:
    break
  

  if current_task == TaskType.UP:
    action = read_from_pipe(action_pipe, 1024)
    # distribute action type
    obs, state, reward, is_terminal, info = step(action)
    # print(state)
    print(action)
    # print(np.array(state[-7:])*2*np.pi)
    '''write_to_pipe([touch_pipe, reward_pipe, terminal_pipe, over_pipe, obs_pipe], [state, reward, is_terminal, done(), obs])'''
    write_to_pipe([touch_pipe, reward_pipe, terminal_pipe, over_pipe], [state, reward, is_terminal, done()])
    rst = read_from_pipe(reset_pipe, 20)
    
    if rst:
      close_pipe([action_pipe,space_pipe,obs_pipe,touch_pipe,reward_pipe,over_pipe,terminal_pipe,reset_pipe,channel_pipe])
      env.environment.set_goal_position('down')
      env.environment.reset_eval_to_down()
      write_to_pipe(touch_pipe_down, env.environment._get_state())
      task_done = True

      # reset()
      # do not reset when demo with eval mode agent
  elif current_task == TaskType.DOWN:
    action = read_from_pipe(action_pipe_down, 1024)
    # distribute action type
    obs, state, reward, is_terminal, info = step(action)
    # print(state)
    print(action)
    # print(np.array(state[-7:])*2*np.pi)
    '''write_to_pipe([touch_pipe, reward_pipe, terminal_pipe, over_pipe, obs_pipe], [state, reward, is_terminal, done(), obs])'''
    write_to_pipe([touch_pipe_down, reward_pipe_down, terminal_pipe_down, over_pipe_down], [state, reward, is_terminal, done()])
    rst = read_from_pipe(reset_pipe_down, 20)
    
    if rst:
      close_pipe([action_pipe_down,space_pipe_down,obs_pipe_down,touch_pipe_down,reward_pipe_down,over_pipe_down,terminal_pipe_down,reset_pipe_down,channel_pipe_down])
      task_done = True
      # reset()
      # do not reset when demo with eval mode agent
  else:


    # IK control may be here

    ##
    # you may adjust positions of motors before dopamine executing downward
    # adjust to init_positions in down.wbt
    ##


    pass

# Enter here exit cleanup code.
