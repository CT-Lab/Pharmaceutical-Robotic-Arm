from deepbots.robots.controllers.robot_emitter_receiver_csv import RobotEmitterReceiverCSV
import numpy as np
def motorToRange(motorPosition, i):
	if(i==0):
		motorPosition = np.clip(motorPosition, -2.897, 2.897)
	elif(i==1):
		motorPosition = np.clip(motorPosition, -1.763, 1.763)
	elif(i==2):
		motorPosition = np.clip(motorPosition, -2.8973, 2.8973)
	elif(i==3):
		motorPosition = np.clip(motorPosition, -3.072, -0.07)
	elif(i==4):
		motorPosition = np.clip(motorPosition, -2.8973, 2.8973)
	elif(i==5):
		motorPosition = np.clip(motorPosition, -0.0175, 3.7525)
	elif(i==6):
		motorPosition = np.clip(motorPosition, -2.897, 2.897)
	else:
		pass
	return motorPosition
class PandaRobot(RobotEmitterReceiverCSV):
	def __init__(self):
		super().__init__()
		self.positionSensorList = []
		for i in range(7):
			positionSensorName = 'positionSensor' + str(i+1)
			positionSensor = self.robot.getPositionSensor(positionSensorName)
			positionSensor.enable(self.get_timestep())
			self.positionSensorList.append(positionSensor)
		self.motorList = []
		for i in range(7):
			motorName = 'motor' + str(i + 1)
			motor = self.robot.getMotor(motorName)	 # Get the motor handle #positionSensor1
			motor.setPosition(float('inf'))  # Set starting position
			motor.setVelocity(0.0)  # Zero out starting velocity
			self.motorList.append(motor)
		motorName = 'finger motor L'
		motor = self.robot.getMotor(motorName)	 # Get the motor handle #positionSensor1
		motor.setPosition(0.02)  # Set starting position
		motor.setVelocity(0.2)  # Zero out starting velocity
		self.motorList.append(motor)
		motorName = 'finger motor R'
		motor = self.robot.getMotor(motorName)	 # Get the motor handle #positionSensor1
		motor.setPosition(0.02)  # Set starting position
		motor.setVelocity(0.2)  # Zero out starting velocity
		self.motorList.append(motor)

		self.initFlag = 0
		self.initPos = []
	def create_message(self):
		# Read the sensor value, convert to string and save it in a list
		message = [str(self.positionSensorList[0].getValue()), str(self.positionSensorList[1].getValue()),\
			str(self.positionSensorList[2].getValue()), str(self.positionSensorList[3].getValue()),\
			str(self.positionSensorList[4].getValue()), str(self.positionSensorList[5].getValue()),\
			str(self.positionSensorList[6].getValue())]
		return message
	
	def use_message_data(self, message):
		#print("robot get this message: ", message)
		code = int(message[0])
		setVelocityList = []
		# decoding action
		# for i in range(7):
		# 	setVelocityList.append(code%3)
		# 	code = int(code/3)
		for i in range(6):
			setVelocityList.append(code%3)
			code = int(code/3)
		#print("decode message to action: ", setVelocityList)

		# # version1 add Velocity
		# for i in range(7):
		# 	action = setVelocityList[i]  # Convert the string message into an action integer
		# 	if action == 2:
		# 		motorSpeed = -1.0
		# 	elif action == 1:
		# 		motorSpeed = 1.0
		# 	else:
		# 		motorSpeed = 0.0
		# 	self.motorList[i].setVelocity(motorSpeed) # Set the motors' velocities based on the action received
		
		# version2 add position
		# for i in range(7):
		# 	action = setVelocityList[i]
		# 	if action == 2:
		# 		motorPosition = self.positionSensorList[i].getValue()-0.05
		# 		motorPosition = motorToRange(motorPosition, i)
		# 		self.motorList[i].setVelocity(2.5)
		# 		self.motorList[i].setPosition(motorPosition) 
		# 	elif action == 1:
		# 		motorPosition = self.positionSensorList[i].getValue()+0.05
		# 		motorPosition = motorToRange(motorPosition, i)
		# 		self.motorList[i].setVelocity(2.5)
		# 		self.motorList[i].setPosition(motorPosition)
		# 	else:
		# 		motorPosition = self.positionSensorList[i].getValue()
		# 		motorPosition = motorToRange(motorPosition, i)
		# 		self.motorList[i].setVelocity(2.5)
		# 		self.motorList[i].setPosition(motorPosition)
		
		for i in range(1, 7):
			action = setVelocityList[i-3]
			if action == 2:
				motorPosition = self.positionSensorList[i].getValue()-0.05
				motorPosition = motorToRange(motorPosition, i)
				self.motorList[i].setVelocity(2.5)
				self.motorList[i].setPosition(motorPosition) 
			elif action == 1:
				motorPosition = self.positionSensorList[i].getValue()+0.05
				motorPosition = motorToRange(motorPosition, i)
				self.motorList[i].setVelocity(2.5)
				self.motorList[i].setPosition(motorPosition)
			else:
				motorPosition = self.positionSensorList[i].getValue()
				motorPosition = motorToRange(motorPosition, i)
				self.motorList[i].setVelocity(2.5)
				self.motorList[i].setPosition(motorPosition)

		if(self.initFlag==0):
			for i in range(1):
				self.initPos.append(self.positionSensorList[i].getValue())
			self.initFlag==1
		for i in range(1):
			self.motorList[i].setVelocity(2.5)
			self.motorList[i].setPosition(self.initPos[i])
# Create the robot controller object and run it
robot_controller = PandaRobot()
robot_controller.run()  # Run method is implemented by the framework, just need to call it