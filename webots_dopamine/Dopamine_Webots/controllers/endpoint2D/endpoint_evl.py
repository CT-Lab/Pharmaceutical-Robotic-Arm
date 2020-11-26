from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



from absl import app
from absl import flags
from agents.dqn import dqn_agent
# from agents.ddpg import ddpg_agent
from agents.ddpg import ddpg_agent_s
from agents.implicit_quantile import implicit_quantile_agent
from agents.rainbow import rainbow_agent
import time
import os
import json
import numpy as np
import csv
# import run_experiment
from gym import spaces
import tensorflow as tf
from pipe import make_pipe, close_pipe, open_write_pipe, open_read_pipe, write_to_pipe, read_from_pipe

# share pipe in every slave
channel_name = "/tmp/channel_in1.pipe"
space_path = "/tmp/space_out1.pipe"
goal_path = "/tmp/goal_in1.pipe"

action_path = "/tmp/action_in1.pipe"
obs_path = "/tmp/obs_out1.pipe"
touch_path = "/tmp/touch_out1.pipe"
reward_path = "/tmp/reward_out1.pipe"
over_path = "/tmp/over_out1.pipe"
terminal_path = "/tmp/term_out1.pipe"
reset_path = "/tmp/reset_in1.pipe"
write_name_list = [action_path, reset_path]
read_name_list = [obs_path, touch_path, reward_path, over_path, terminal_path]

channel_pipe = open_write_pipe(channel_name)
write_to_pipe(channel_pipe, 0)
complete_pipe = open_write_pipe("/tmp/complete.pipe")
write_to_pipe(complete_pipe, 0)
goal_pipe = open_write_pipe(goal_path)

agent_name = 'ddpg'
debug_mode = False

def create_agent(sess, summary_writer=None):

  # s = os.open(space_path, os.O_RDONLY)
  s = open_read_pipe(space_path)
  # space = json.loads(os.read(s,1024).decode())
  space = read_from_pipe(s)
  close_pipe([channel_pipe, complete_pipe])
  if not debug_mode:
    summary_writer = None
  if agent_name == 'ddpg':
    os.close(s)
    return ddpg_agent_s.DDPGAgent(sess, action_space=spaces.Box(space[0], space[1], shape=space[2], dtype=np.float32), #num_actions=space,
                              summary_writer=summary_writer)
  elif agent_name == 'dqn':
    os.close(s)
    return dqn_agent.DQNAgent(sess, num_actions=space,
                              summary_writer=summary_writer)
  elif agent_name == 'rainbow':
    os.close(s)
    return rainbow_agent.RainbowAgent(
        sess, num_actions=space,
        summary_writer=summary_writer)
  elif agent_name == 'implicit_quantile':
    os.close(s)
    return implicit_quantile_agent.ImplicitQuantileAgent(
        sess, num_actions=space,
        summary_writer=summary_writer)
  else:
    os.close(s)
    raise ValueError('Unknown agent: {}'.format(agent_name))

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    
    agent = create_agent(sess)
    agent.eval_mode = True
    # length = round(np.random.uniform(0.5, 0.7), 4)
    # theta = round(np.random.uniform(0, np.pi/2), 4)
    length = np.linspace(0.53, 0.7, 5)
    theta = np.linspace(0.3, np.pi/2, 20)
    statistic = []
    # filename = 'statistic/DDPG_mlit_4s_v0508_3w30_rndinit.csv'
    # outfile = open(filename, 'w', newline='')
    # Run every checkpoint
    for i in range(10):
        filename = 'statistic/test_wait/test_wait%d.csv'%i
        outfile = open(filename, 'w', newline='')

        agent._saver.restore(sess, "model/test_wait/checkpoints/tf_ckpt-%d"%i)
        ckpt0 = []
        ckpt1 = []
        ckpt2 = []
        print('iterations %d, traning step %d'%(i,(i+1)*30000))
        # Run 100 episodes
        for j in range(100):
            l = int(j/20)
            t = j%20

            total_reward = 0
            is_terminal = False
            channel_pipe = open_write_pipe(channel_name)
            write_to_pipe(channel_pipe, 0)
            complete_pipe = open_write_pipe("/tmp/complete.pipe")
            write_to_pipe(complete_pipe, 1)

            write_to_pipe(goal_pipe, [round(length[l],4), round(theta[t], 4)])

            action_pipe, reset_pipe = open_write_pipe(write_name_list)
            obs_pipe, touch_pipe, reward_pipe, over_pipe, terminal_pipe = open_read_pipe(read_name_list)
            """initial_observation_list = read_from_pipe(obs_pipe)"""
            """initial_observation = np.asarray(initial_observation_list)"""
            initial_observation = np.zeros([100,100,3])
            initial_state_list = read_from_pipe(touch_pipe)
            initial_state = np.asarray(initial_state_list)
            action = agent.begin_episode(initial_observation, initial_state)
            time.sleep(0.032)
            print('episodes %d'%j)
            episode_distance = []
            cnt1 = 0
            cnt2 = 0
            step_cnt = 0
            while 1:
                action = action.tolist()
                write_to_pipe(action_pipe, action)
                
                state = read_from_pipe(touch_pipe)
                state = np.asarray(state)
                reward = read_from_pipe(reward_pipe)
                # reward = np.clip(reward, -1, 1)
                is_terminal = read_from_pipe(terminal_pipe)
                # print('distance', reward)
                episode_distance.append(reward)
                if reward < 0.02:
                    cnt1 = 1
                    cnt2 += 1
                    if cnt2 == 20:
                        is_terminal = True
                else:
                    cnt2 = 0

                step_cnt += 1
                if step_cnt == 400:
                    is_terminal = True

                """observation = read_from_pipe(obs_pipe)
                observation = np.asarray(observation)"""
                observation = np.zeros([100,100,3])
                over = read_from_pipe(over_pipe)
                over = over or is_terminal
                # os.write(reset_p, json.dumps(is_terminal).encode())
                write_to_pipe(reset_pipe, over)
                if over:
                    close_pipe([obs_pipe, touch_pipe, reward_pipe, over_pipe, terminal_pipe, action_pipe, reset_pipe, channel_pipe, complete_pipe])
                    # print("broken")
                    time.sleep(0.032)
                    break
                else:
                    action = agent.step(reward, observation, state)
            
            if np.mean(episode_distance[-20:]) < 0.01:
                ckpt0.append(1)
            elif np.mean(episode_distance[-20:]) < 0.02:
                ckpt0.append(2)
            else:
                ckpt0.append(0)
            # if cnt == 20:
            if cnt1:
                ckpt1.append(1)
            else:
                ckpt1.append(0)
            
            if cnt2 == 20:
                ckpt2.append(1)
            else:
                ckpt2.append(0)
        succ_rate = np.sum(np.array(ckpt0)>0)/len(ckpt0)
        ckpt0.insert(0, succ_rate)

        succ_rate = np.sum(np.array(ckpt1)>0)/len(ckpt1)
        ckpt1.insert(0, succ_rate)

        succ_rate = np.sum(np.array(ckpt2)>0)/len(ckpt2)
        ckpt2.insert(0, succ_rate)

        writer = csv.writer(outfile)
        writer.writerow(ckpt0)
        writer.writerow(ckpt1)
        writer.writerow(ckpt2)
        outfile.close()