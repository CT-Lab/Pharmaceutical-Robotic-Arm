import circular_replay_buffer
import numpy as np
import tensorflow as tf
import cv2

observation_shape = (100,100,3)
state_shape = (3,)
action_dim = (2,)
stack_size = 1
use_staging = False
update_horizon = 1
gamma = 0.98
observation_dtype = tf.uint8
state_dtype = np.float32
sess = tf.Session()

m = circular_replay_buffer.WrappedReplayBuffer(
    observation_shape=observation_shape,
	state_shape=state_shape,
	action_dim=action_dim,
	stack_size=stack_size,
	use_staging=use_staging,
	update_horizon=update_horizon,
	gamma=gamma,
	observation_dtype=observation_dtype.as_numpy_dtype,
	state_dtype=state_dtype)


mm = m.memory

m.add(np.ones((100,100,3))*255,np.arange(3), np.arange(2), 1, 0)
m.add(np.ones((100,100,3))*200,np.arange(3)*2, np.arange(2)*2, 1, 0)
m.add(np.ones((100,100,3))*150,np.arange(3)*3, np.arange(2)*3, 1, 0)
m.add(np.ones((100,100,3))*100,np.arange(3)*4, np.arange(2)*4, 1, 0)
m.add(np.ones((100,100,3))*50,np.arange(3)*5, np.arange(2)*5, 1, 0)
m.add(np.ones((100,100,3))*0,np.arange(3)*6, np.arange(2)*6, 1, 0)
print(m.observations)

a=sess.run(m.observations)
b=sess.run([m.states,m.next_states])
print(np.hstack([b[0],b[1]]))
b=sess.run([m.states,m.next_states])
print(np.hstack([b[0],b[1]]))
b=sess.run([m.states,m.next_states])
print(np.hstack([b[0],b[1]]))


target=m.states
replay=m.next_states

d = replay-target
print(sess.run([d, target, m.actions]))
# print((b+np.array([0,1,2])==c))
# c=sess.run(m.next_states)
# d=(b==(c+np.array[0,1,2]))
# print(np.hstack([b,c]))
print(m.actions)
