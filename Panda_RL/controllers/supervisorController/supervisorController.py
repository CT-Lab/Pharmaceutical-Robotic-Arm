import numpy as np
from deepbots.supervisor.controllers.supervisor_emitter_receiver import SupervisorCSV
from PPOAgent import PPOAgent, Transition
from utilities import normalizeToRange
from ArmUtil import ToArmCoord, PSFunc
import random as rand
from DDPGAgent import DDPGAgent
from controller import Keyboard

class PandaSupervisor(SupervisorCSV):
	def __init__(self):
		super().__init__()  
		#self.observationSpace = 14  # The agent has 7 inputs 
		self.observationSpace = 13 # no motorposition 0, 1, and 2
		#self.actionSpace = 2187 # The agent can perform 3^7 actions
		self.actionSpace = 729 # 3^6

		self.robot = None
		self.beaker = None
		self.target = None
		self.target = self.supervisor.getFromDef("TARGET")
		print(type(self.target))
		self.endEffector = None 
		# self.armChain = Chain.from_urdf_file("panda_with_bound.URDF")
		self.respawnRobot()
		self.messageReceived = None	 # Variable to save the messages received from the robot
		
		self.episodeCount = 0  # Episode counter
		self.episodeLimit = 20000  # Max number of episodes allowed
		self.stepsPerEpisode = 100  # Max number of steps per episode
		self.episodeScore = 0  # Score accumulated during an episode
		self.episodeScoreList = []  # A list to save all the episode scores, used to check if task is solved
		self.doneTotal = 0

		self.CAM_A = self.supervisor.getCamera("CAM_A")
		self.CAM_B = self.supervisor.getCamera("CAM_B")
		self.CAM_C = self.supervisor.getCamera("CAM_C")
		self.CAM_D = self.supervisor.getCamera("CAM_D")

		self.CAM_A.enable(32)
		self.CAM_B.enable(32)
		self.CAM_C.enable(32)
		self.CAM_D.enable(32)

		self.CAM_A.recognitionEnable(32)
		self.CAM_B.recognitionEnable(32)
		self.CAM_C.recognitionEnable(32)
		self.CAM_D.recognitionEnable(32)
		
		self.Linear_motor_CAM_A = self.supervisor.getMotor("linear_motor_CAM_A")
		self.Linear_motor_CAM_A.setPosition(float('inf'))  # Set starting position
		self.Linear_motor_CAM_A.setVelocity(0.0)  # Zero out starting velocity
		self.Linear_positionSensor_CAM_A = self.supervisor.getPositionSensor("linear_positionSensor_CAM_A")
		self.Linear_positionSensor_CAM_A.enable(int(self.supervisor.getBasicTimeStep()))
		
		self.Rot_motor_CAM_A = self.supervisor.getMotor("rotational_motor_CAM_A")
		self.Rot_motor_CAM_A.setPosition(float('inf'))  # Set starting position
		self.Rot_motor_CAM_A.setVelocity(0.0)  # Zero out starting velocity
		self.Rot_motor_positionSensor_CAM_A = self.supervisor.getPositionSensor("rotational_positionSensor_CAM_A")
		self.Rot_motor_positionSensor_CAM_A.enable(int(self.supervisor.getBasicTimeStep()))

		self.Linear_motor_CAM_B = self.supervisor.getMotor("linear_motor_CAM_B")
		self.Linear_motor_CAM_B.setPosition(float('inf'))  # Set starting position
		self.Linear_motor_CAM_B.setVelocity(0.0)  # Zero out starting velocity
		self.Linear_positionSensor_CAM_B = self.supervisor.getPositionSensor("linear_positionSensor_CAM_B")
		self.Linear_positionSensor_CAM_B.enable(int(self.supervisor.getBasicTimeStep()))
		
		self.Rot_motor_CAM_B = self.supervisor.getMotor("rotational_motor_CAM_B")
		self.Rot_motor_CAM_B.setPosition(float('inf'))  # Set starting position
		self.Rot_motor_CAM_B.setVelocity(0.0)  # Zero out starting velocity
		self.Rot_motor_positionSensor_CAM_B = self.supervisor.getPositionSensor("rotational_positionSensor_CAM_B")
		self.Rot_motor_positionSensor_CAM_B.enable(int(self.supervisor.getBasicTimeStep()))

		self.Linear_motor_CAM_C = self.supervisor.getMotor("linear_motor_CAM_C")
		self.Linear_motor_CAM_C.setPosition(float('inf'))  # Set starting position
		self.Linear_motor_CAM_C.setVelocity(0.0)  # Zero out starting velocity
		self.Linear_positionSensor_CAM_C = self.supervisor.getPositionSensor("linear_positionSensor_CAM_C")
		self.Linear_positionSensor_CAM_C.enable(int(self.supervisor.getBasicTimeStep()))
		
		self.Rot_motor_CAM_C = self.supervisor.getMotor("rotational_motor_CAM_C")
		self.Rot_motor_CAM_C.setPosition(float('inf'))  # Set starting position
		self.Rot_motor_CAM_C.setVelocity(0.0)  # Zero out starting velocity
		self.Rot_motor_positionSensor_CAM_C = self.supervisor.getPositionSensor("rotational_positionSensor_CAM_C")
		self.Rot_motor_positionSensor_CAM_C.enable(int(self.supervisor.getBasicTimeStep()))

		self.Linear_motor_CAM_D = self.supervisor.getMotor("linear_motor_CAM_D")
		self.Linear_motor_CAM_D.setPosition(float('inf'))  # Set starting position
		self.Linear_motor_CAM_D.setVelocity(0.0)  # Zero out starting velocity
		self.Linear_positionSensor_CAM_D = self.supervisor.getPositionSensor("linear_positionSensor_CAM_D")
		self.Linear_positionSensor_CAM_D.enable(int(self.supervisor.getBasicTimeStep()))
		
		self.Rot_motor_CAM_D = self.supervisor.getMotor("rotational_motor_CAM_D")
		self.Rot_motor_CAM_D.setPosition(float('inf'))  # Set starting position
		self.Rot_motor_CAM_D.setVelocity(0.0)  # Zero out starting velocity
		self.Rot_motor_positionSensor_CAM_D = self.supervisor.getPositionSensor("rotational_positionSensor_CAM_D")
		self.Rot_motor_positionSensor_CAM_D.enable(int(self.supervisor.getBasicTimeStep()))

		self.keyboard = self.supervisor.getKeyboard()
		self.keyboard.enable(int(self.supervisor.getBasicTimeStep()))

	def respawnRobot(self):
		if self.robot is not None:
			# Despawn existing robot
			self.robot.remove()
			self.beaker.remove()

		# Respawn robot in starting position and state
		rootNode = self.supervisor.getRoot()  # This gets the root of the scene tree
		childrenField = rootNode.getField('children')  # This gets a list of all the children, ie. objects of the scene
		childrenField.importMFNode(-2, "Solid.wbo")	 # Load Beaker from file and add to second-to-last position
		childrenField.importMFNode(-2, "Robot.wbo")	 # Load robot from file and add to second-to-last position
		# Get the new robot 
		self.beaker = self.supervisor.getFromDef("Beaker")
		self.robot = self.supervisor.getFromDef("Franka_Emika_Panda")
		self.endEffector = self.supervisor.getFromDef("endEffector")

		targetPosition = ToArmCoord.convert(self.target.getPosition()) # transfer to arm coordinate system
		endEffectorPosition = ToArmCoord.convert(self.endEffector.getPosition()) # transfer to arm coordinate system
		self.preL2norm = np.linalg.norm([targetPosition[0]-endEffectorPosition[0],targetPosition[1]-endEffectorPosition[1],targetPosition[2]-endEffectorPosition[2]])
		# print("test:",self.endEffector.getPosition())
	def get_observations(self): # [motorPosition, targetPosition, endEffectorPosition, L2norm(targetPosition vs endEffectorPosition)]
		# Update self.messageReceived received from robot, which contains motor position
		self.messageReceived = self.handle_receiver()
		if self.messageReceived is not None:
			# for 7 motors
			# motorPosition = [float(self.messageReceived[0]), float(self.messageReceived[1]), float(self.messageReceived[2]), \
			# 	float(self.messageReceived[3]), float(self.messageReceived[4]), float(self.messageReceived[5]), \
			# 	float(self.messageReceived[6])]
			# for 4 motors
			motorPosition = [float(self.messageReceived[1]),float(self.messageReceived[2]),float(self.messageReceived[3]), float(self.messageReceived[4]), float(self.messageReceived[5]), \
				float(self.messageReceived[6])]
		else:
			# Method is called before self.messageReceived is initialized
			# motorPosition = [0.0 for _ in range(7)]
			# motorPosition[3] = -0.0698
			motorPosition = [0.0 for _ in range(6)]
			motorPosition[0] = -0.0698
		
		# get TARGET posion
		targetPosition = self.target.getPosition()
		targetPosition = ToArmCoord.convert(targetPosition) # transfer to arm coordinate system
		
		if self.messageReceived is not None:
			# get end-effort position
			motorPosition_for_FK = motorPosition + [0.0]
			# endEffectorPosition = self.armChain.forward_kinematics(motorPosition_for_FK)[0:3, 3] # This is already in arm coordinate.
			endEffectorPosition = ToArmCoord.convert(self.endEffector.getPosition()) # transfer to arm coordinate system
			# compute L2 norm
			# print("[Debug tartgetPosition]:",targetPosition)
			# print("[Debug endEffectorPosition]:",endEffectorPosition)
			L2norm = np.linalg.norm([targetPosition[0]-endEffectorPosition[0],targetPosition[1]-endEffectorPosition[1],targetPosition[2]-endEffectorPosition[2]])
		else:
			# tmp = [0.0 for _ in range(8)]
			# tmp[3] = -0.0698
			# endEffectorPosition = self.armChain.forward_kinematics(tmp)[0:3, 3] # This is already in arm coordinate.
			endEffectorPosition = ToArmCoord.convert(self.endEffector.getPosition()) # transfer to arm coordinate system
			L2norm = self.preL2norm
		
		# convert to a single list
		returnObservation = [*motorPosition, *targetPosition, *endEffectorPosition, L2norm] # 6+3+3+1 = 13
		return returnObservation
		
	def get_reward(self, action=None):
		return 0 # I implement in the comment:'compute reward here' 
	
	def is_done(self):
		return False
	
	def solved(self):
		return False
		if len(self.episodeScoreList) > 1000:  # Over 100 trials thus far
			if np.mean(self.episodeScoreList[-100:]) > 1000:  # Last 100 episodes' scores average value
				return True
		return False
		
	def reset(self):
		self.respawnRobot()
		self.supervisor.simulationResetPhysics()  # Reset the simulation physics to start over
		self.messageReceived = None
		return [0.0 for _ in range(self.observationSpace)]
		
	def get_info(self):
		return "I'm trying to reach that red ball!"


# robot = Robot()
# camera = robot.getCamera("CAM_A")
# camera.enable(32)

supervisor = PandaSupervisor()
# agent = PPOAgent(supervisor.observationSpace, supervisor.actionSpace, use_cuda=False) #add use_cuda
agent = DDPGAgent(alpha=0.000025, beta=0.00025, input_dims=supervisor.observationSpace, tau=0.001,
              batch_size=100,  n_actions=supervisor.actionSpace)
# supervisor.camera.enable(32)

solved = False
cnt_veryClose = 0
# Run outer loop until the episodes limit is reached or the task is solved
while not solved and supervisor.episodeCount < supervisor.episodeLimit:
	observation = supervisor.reset()  # Reset robot and get starting observation
	supervisor.episodeScore = 0
	cnt_veryClose = 0
	NEAR_DISTANCE = 0.05
	BONUS_REWARD = 10
	ON_GOAL_FINISH_COUNT = 5
	left_bonus_count = ON_GOAL_FINISH_COUNT
	for step in range(supervisor.stepsPerEpisode):
		# get keyboard input
		keyboard=supervisor.keyboard
		key=keyboard.getKey()
		# camera control-----------------------
		if (key==ord('1')):
			print('CAM_A rot:left')
			supervisor.Rot_motor_CAM_A.setVelocity(1.0)
			supervisor.Rot_motor_CAM_A.setPosition(supervisor.Rot_motor_positionSensor_CAM_A.getValue()+0.05)
		elif (key==ord('Q')):
			print('CAM_A rot:right')
			supervisor.Rot_motor_CAM_A.setVelocity(1.0)
			supervisor.Rot_motor_CAM_A.setPosition(supervisor.Rot_motor_positionSensor_CAM_A.getValue()-0.05)
		elif (key==ord('2')):
			print('CAM_B rot:left')
			supervisor.Rot_motor_CAM_B.setVelocity(1.0)
			supervisor.Rot_motor_CAM_B.setPosition(supervisor.Rot_motor_positionSensor_CAM_B.getValue()+0.05)
		elif (key==ord('W')):
			print('CAM_B rot:right')
			supervisor.Rot_motor_CAM_B.setVelocity(1.0)
			supervisor.Rot_motor_CAM_B.setPosition(supervisor.Rot_motor_positionSensor_CAM_B.getValue()-0.05)
		elif (key==ord('3')):
			print('CAM_C rot:left')
			supervisor.Rot_motor_CAM_C.setVelocity(1.0)
			supervisor.Rot_motor_CAM_C.setPosition(supervisor.Rot_motor_positionSensor_CAM_C.getValue()+0.05)
		elif (key==ord('E')):
			print('CAM_C rot:right')
			supervisor.Rot_motor_CAM_C.setVelocity(1.0)
			supervisor.Rot_motor_CAM_C.setPosition(supervisor.Rot_motor_positionSensor_CAM_C.getValue()-0.05)
		elif (key==ord('4')):
			print('CAM_D rot:left')
			supervisor.Rot_motor_CAM_D.setVelocity(1.0)
			supervisor.Rot_motor_CAM_D.setPosition(supervisor.Rot_motor_positionSensor_CAM_D.getValue()+0.05)
		elif (key==ord('R')):
			print('CAM_D rot:right')
			supervisor.Rot_motor_CAM_D.setVelocity(1.0)
			supervisor.Rot_motor_CAM_D.setPosition(supervisor.Rot_motor_positionSensor_CAM_D.getValue()-0.05)
		elif (key==ord('5')):
			print('ALL higher')
			supervisor.Linear_motor_CAM_A.setVelocity(1.0)
			supervisor.Linear_motor_CAM_B.setVelocity(1.0)
			supervisor.Linear_motor_CAM_C.setVelocity(1.0)
			supervisor.Linear_motor_CAM_D.setVelocity(1.0)
			supervisor.Linear_motor_CAM_A.setPosition(supervisor.Linear_positionSensor_CAM_A.getValue()+0.05)
			supervisor.Linear_motor_CAM_B.setPosition(supervisor.Linear_positionSensor_CAM_B.getValue()+0.05)
			supervisor.Linear_motor_CAM_C.setPosition(supervisor.Linear_positionSensor_CAM_C.getValue()+0.05)
			supervisor.Linear_motor_CAM_D.setPosition(supervisor.Linear_positionSensor_CAM_D.getValue()+0.05)
		elif (key==ord('T')):
			print('ALL lowerer')
			supervisor.Linear_motor_CAM_A.setVelocity(1.0)
			supervisor.Linear_motor_CAM_B.setVelocity(1.0)
			supervisor.Linear_motor_CAM_C.setVelocity(1.0)
			supervisor.Linear_motor_CAM_D.setVelocity(1.0)
			supervisor.Linear_motor_CAM_A.setPosition(supervisor.Linear_positionSensor_CAM_A.getValue()-0.05)
			supervisor.Linear_motor_CAM_B.setPosition(supervisor.Linear_positionSensor_CAM_B.getValue()-0.05)
			supervisor.Linear_motor_CAM_C.setPosition(supervisor.Linear_positionSensor_CAM_C.getValue()-0.05)
			supervisor.Linear_motor_CAM_D.setPosition(supervisor.Linear_positionSensor_CAM_D.getValue()-0.05)
		else:
			pass
		# # camera control-----------------------end
		# if(supervisor.CAM_A.getRecognitionObjects()!= []):
		# 	position = supervisor.CAM_A.getRecognitionObjects()[0].get_position_on_image()
		# 	print("position_on_image:", position)
		# 	size = supervisor.CAM_A.getRecognitionObjects()[0].get_size_on_image()
		# 	print("size_on_image:", size)
		
		# In training mode the agent samples from the probability distribution, naturally implementing exploration
		# selectedAction, actionProb = agent.work(observation, type_="selectAction")
		selectedAction = agent.choose_action(observation)
		actionProb = np.amax(selectedAction)
		selectedAction=np.where(selectedAction == np.amax(selectedAction))[0][0]
		# print(selectedAction, actionProb)
		# rand.seed()
		# print("ActionProb:",actionProb)
		# if(rand.randint(1,10)==1):
		# 	rand.seed()
		# 	# print("Origin Master: ", selectedAction)
		# 	selectedAction = rand.randint(0,2187-1)
		# 	# print("Change Action: ", selectedAction)
		# Step the supervisor to get the current selectedAction's reward, the new observation and whether we reached 
		# the done condition
		newObservation, reward, done, info = supervisor.step([selectedAction])
		# print("L2",newObservation[-1])
		# compute done here
		# if newObservation[-1] <= 0.01:
		# 	cnt_veryClose += 1
		# if cnt_veryClose >= 50 or step==supervisor.stepsPerEpisode-1:
		# 	done = True
		# 	supervisor.preL2norm=0.4
		# compute reward here
		## do not get too close to the limit value 
		# [-2.897, 2.897], [-1.763, 1.763], [-2.8973, 2.8973], [-3.072, -0.0698]
		# [-2.8973, 2.8973], [-0.0175, 3.7525], [-2.897, 2.897]
		# if newObservation[0]-(-2.897)<0.05 or 2.897-newObservation[0]<0.05 or\
		# 	newObservation[1]-(-1.763)<0.05 or 1.763-newObservation[1]<0.05 or\
		# 	newObservation[2]-(-2.8973)<0.05 or 2.8973-newObservation[2]<0.05 or\
		# 	newObservation[3]-(-3.072)<0.05 or -0.0697976-newObservation[3]<0.05 or\
		# 	newObservation[4]-(-2.8973)<0.05 or 2.8973-newObservation[4]<0.05 or\
		# 	newObservation[5]-(-0.0175)<0.05 or 3.7525-newObservation[5]<0.05 or\
		# 	newObservation[6]-(-2.897)<0.05 or 2.897-newObservation[6]<0.05:
		# 	reward = -1 # if on of the motors on the limit, reward = -2
		# else:
		# 	if(newObservation[-1]<0.01):
		# 		reward = 10 #*((supervisor.stepsPerEpisode - step)/supervisor.stepsPerEpisode) 
		# 	elif(newObservation[-1]<0.05):
		# 		reward = 5 #*((supervisor.stepsPerEpisode - step)/supervisor.stepsPerEpisode)
		# 	elif(newObservation[-1]<0.1):
		# 		reward = 1 #*((supervisor.stepsPerEpisode - step)/supervisor.stepsPerEpisode)
		# 	else:
		# 		reward = -(newObservation[-1]-supervisor.preL2norm) # positive reward
		# 	supervisor.preL2norm = newObservation[-1]
		# print("Beaker: ",supervisor.beaker.getOrientation(),"=>",supervisor.beaker.getOrientation()[4])
		# reward = reward + 0.05*(supervisor.beaker.getOrientation()[4] - 1.0) # We want it to remain horizontal.
		reward = supervisor.preL2norm-newObservation[-1]
		
		supervisor.preL2norm = newObservation[-1]
		# compute done here
		if newObservation[-1] < NEAR_DISTANCE:
			cnt_veryClose += 1
			if left_bonus_count > 0:
				reward += BONUS_REWARD*(supervisor.stepsPerEpisode-step)/supervisor.stepsPerEpisode
			if cnt_veryClose >= ON_GOAL_FINISH_COUNT:
				done = True
		if step==supervisor.stepsPerEpisode-1:
			done = True
		
		
		reward += 1000*(supervisor.beaker.getOrientation()[4] - 0.001) # We want it to remain horizontal.
		# print("reward: ",reward)
		# print("L2norm: ", newObservation[-1])
		# print("tarPosition(trans): ", newObservation[7:10])
		# print("endPosition: ", newObservation[10:13])
		#print("endPosition(trans): ", ToArmCoord.convert(newObservation[10:13]))
		# ------compute reward end------
		# Save the current state transition in agent's memory
		# trans = Transition(observation, selectedAction, actionProb, reward, newObservation)
		# agent.storeTransition(trans)

		agent.remember(observation, selectedAction, reward, newObservation, int(done))
		if done:
			if(step==0):
				print("0 Step but done?")
				continue
			print("done gogo")
			# Save the episode's score
			supervisor.episodeScoreList.append(supervisor.episodeScore)
			agent.trainStep()
			agent.save_models() # save ddpg models
			solved = supervisor.solved()  # Check whether the task is solved
			# agent.save('')
			break
		
		supervisor.episodeScore += reward  # Accumulate episode reward
		observation = newObservation  # observation for next step is current step's newObservation
		
	fp = open("Episode-score.txt","a")
	fp.write(str(supervisor.episodeScore)+'\n')
	fp.close()
	print("Episode #", supervisor.episodeCount, "score:", supervisor.episodeScore)
	supervisor.episodeCount += 1  # Increment episode counter

if not solved:
	print("Task is not solved, deploying agent for testing...")
elif solved:
	print("Task is solved, deploying agent for testing...")
	
observation = supervisor.reset()

while True:
	selectedAction, actionProb = agent.work(observation, type_="selectActionMax")
	observation, _, _, _ = supervisor.step([selectedAction])
