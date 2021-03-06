from controller import PositionSensor, Motor
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
from ArmUtil import ToArmCoord, PSFunc
import numpy as np


class PandaController:
    """
        負責直接控制手臂，也就是呼叫webots function的部分
    """

    #############################
    # TODO: 統一 script, action #
    #############################

    def __init__(self, motorList, psList=None, urdf_file_path="../../panda_with_bound.urdf"):
        """
            @param
                motorList: 各關節的驅動器。
                psList: position sensor list，各關節的角度偵測器
                urdf_file_path: 手臂的URDF檔案位置
                注意，以上參數的關節個數必須相同
        """
        self.motorList = motorList
        self.psList = psList
        self.armChain = Chain.from_urdf_file(urdf_file_path)
        self.script = None

    def assign(self, script=None, action=None):
        """
            指定要執行的腳本(Script)或是動作(Action)
            @param
                script: 一個script物件，要執行的腳本
                action: 單一個action物件。目前只測試過Actuate
        """
        if (type(script) == type(None) and type(action) == type(None)):
            raise Exception("Must assign a script or an action to execute.")
        if (type(script) != type(None) and type(action) != type(None)):
            raise Exception("Only a script or an action can be assigned at once.")
        
        if (type(script) != type(None)):
            self.script = script
            self.script._initialize(self.armChain, self.psList)
        elif (type(action) != type(None)):
            self.action = action

    def step(self, **kwargs):
        """
            執行腳本/動作
        """
        if (type(self.script) != type(None)):
            results = self.script.step(**kwargs)
        elif(type(self.action) != type(None)):
            results = self.action.step(**kwargs)

        if (type(results) == type(None)):
            return
        elif (type(results) == str and results == "Pause"):
            return
        else:
            for i in range(len(results)):
                self.motorList[i].setPosition(results[i])

    def done(self):
        """
            是否已完成腳本/動作
        """
        if (type(self.script) != type(None)):
            return self.script.done()
        elif (type(self.action) != type(None)):
            return self.action.done()

# TODO
class Action:
    def __init__(self):
        pass
    
    def _is_initialized(self):
        pass

    def step(self):
        pass

    def done(self):
        pass

class Pause(Action):
    """
    實作方法是，回傳"Pause"字串，由PandaController判斷
    """
    def __init__(self, step_count=1):
        """
        step_count:  暫停多少step
        """
        self.step_count = step_count
        self.psValue = None
        self.psList = None
    
    def _initialize(self):
        return

    def _is_initialized(self):
        return True

    def step(self):
        self.step_count -= 1
        print("paused")
        return "Pause"
        
    def done(self):
        return self.step_count <= 0

class Actuate(Action):
    """
    使手臂各關節正轉/逆轉
    NOTE!
    由於我們的手臂關節有速度限制，一次正轉逆轉可能沒辦法在一個step完成，
    那，是否要給他完整轉完指定的角度，才算done?

    ((應該要...

    NOTE NOTE 
    BUT!
    RL訓練、使用的方式，不也是這樣? --
    在這一個step觀察各軸角度 --> 餵給網路 --> 吐出各軸怎麼轉 --> webots執行這一個step。手臂一樣不一定到達了預期的位置。
    但是，網路也不需要等手臂抵達預定位置，也可以算出接下來應該怎麼轉。
    """
    def __init__(self, distance_limit_list=None, step_distance_list=None, step_count=1, direction_list=None, tolerance=0.01):
        """
            使用方法(1): 適用於自行一次一次呼叫step，不是塞在腳本執行時 (主要方法)
            使用方法(2): 塞在腳本中；此時需要先傳進各軸要轉的方向

            distance_limit_list: tuple的陣列；每個關節的角度最大最小值
            step_distance_list: 數值陣列；每一個step要走多少。執行step()的時候傳入的direction_list會乘上這個值作為移動的目標角度
            以上兩者的預設值為依據

            step_count:  For(1).  Actuate了幾次才算done.
            direction_list:  For (2).  各軸要轉動的方向，以及程度。如果這個Actuate是要放在腳本裡的，init時就要給值，後面step就會用他。
        """

        if (type(distance_limit_list) == type(None)):
            distance_limit_list = [
                    (-2.8973, 2.8973),
                    (-1.7628, 1.7628),
                    (-2.8973, 2.8973),
                    (-3.0718, -0.0698),
                    (-2.8973, 2.8973),
                    (-0.0175, 3.7525),
                    (-2.8973, 2.8973)
                ]
        if (type(step_distance_list) == type(None)):
            step_distance_list = [
                    abs(-2.8973 - 2.8973),
                    abs(-1.7628 - 1.7628),
                    abs(-2.8973 - 2.8973),
                    abs(-3.0718 - -0.0698),
                    abs(-2.8973 - 2.8973),
                    abs(-0.0175 - 3.7525),
                    abs(-2.8973 - 2.8973)
                ]

        self.distance_limit_list = distance_limit_list
        self.step_distance_list = step_distance_list
        self.step_count = step_count
        self.direction_list = direction_list
        self.psList = None
        self.starting_position = None
        self.tolerance = np.array([tolerance]*7)
        print("initialized actuate with step_distance_list:")
        print(self.step_distance_list)
        """
            panda的URDF檔(參考官方文件)每個關節的角度限制:
            [
            (-2.8973, 2.8973),
            (-1.7628, 1.7628),
            (-2.8973, 2.8973),
            (-3.0718, -0.0698),
            (-2.8973, 2.8973),
            (-0.0175, 3.7525),
            (-2.8973, 2.8973)
            ]
        """
    
    def _initialize(self, psList):
        self.psList = psList
        self.starting_position = np.array(PSFunc.getValue(self.psList))
        self.goal_position = []
        if (len(self.step_distance_list) == len(self.direction_list)):
            for i in range(len(self.direction_list)):
                # There's probably a better way for this...
                raw_val = self.starting_position[i] + self.step_distance_list[i] * self.direction_list[i]
                if (raw_val > max(self.distance_limit_list[i])):
                    raw_val = max(self.distance_limit_list[i])
                if (raw_val < min(self.distance_limit_list[i])):
                    raw_val = min(self.distance_limit_list[i])
                self.goal_position.append(raw_val)

    def _is_initialized(self):
        return type(self.psList) != type(None) and type(self.starting_position) != type(None)

    def step(self, direction_list=None, psValue=None, **kwargs):
        """
            direction_list:  For (1).  各軸要轉動的方向，以及程度
            psValue:  For (1).  各軸目前的角度。一次一次呼叫Actuate才需要 
                    (因為移動的方法是使用setPosition，需要絕對的角度，所以用目前的角度加上delta來求得目標)
        """
        if (type(None) != type(self.direction_list)):
            # => 這個動作是放在腳本裡的
            direction_list = self.direction_list
            psValue = self.starting_position

        # 否則是以一步一步的方式使用，那麼就需要direction_list與psValue
        if (type(direction_list) == type(None) or type(psValue) == type(None)):
            raise Exception("For step by step use of Actuate, direction_list is needed.  And either psValue or psList")

        position_array = []
        if (len(self.step_distance_list) == len(direction_list)):
            for i in range(len(direction_list)):
                # There's probably a better way for this...
                raw_val = psValue[i] + self.step_distance_list[i] * direction_list[i]
                if (raw_val > max(self.distance_limit_list[i])):
                    raw_val = max(self.distance_limit_list[i])
                if (raw_val < min(self.distance_limit_list[i])):
                    raw_val = min(self.distance_limit_list[i])
                position_array.append(raw_val)

        self.step_count -= 1
        return position_array

    def done(self):
        """
            目前定義為，step()過step_count次數，就算done

            這很有問題XD
            由於有速度限制，step過後很可能沒有移動到想要的角度
            TODO!!
        """
        if (type(None) != type(self.direction_list)):
            # 這是在腳本裡面
            #print("In script")
            
            current_position = PSFunc.getValue(self.psList)
            
            
            
            diff = np.array(self.goal_position) - np.array(current_position)
            
            #print("starting position ", self.starting_position)
            #print("current position  ", psValue)
            #print("goal position     ", goal_position)
            #print("diff              ", diff)
            print(np.abs(diff) < self.tolerance)
            if ((np.abs(diff) < self.tolerance).all()):
                return True
            else:
                return False
        # 這是一次一次執行
        return self.step_count <= 0

class IKReachTarget(Action):
    """
        利用IKpy移動到指定點
    """
    def __init__(self, target_position, orientation=None, lazyIK=False, tolerance=0.05, AXIS='Z', o_tolerance=0.2):
        """
            @param
                target_position: [x,y,z]，目標點相對位置
                orientation: [x,y,z]，限制end effector朝向
                tolerance: float，距離目標幾公尺就算作到達目標，預設0.05公尺
                AXIS: 'X'/'Y'/'Z'，在指定orientation時所需，將以此坐標軸對齊orientation所給的方向
        """
        self.target_position = target_position
        self.orientation = orientation
        self.lazyIK = lazyIK
        self.tolerance = tolerance
        self.o_tolerance = o_tolerance
        self.AXIS = AXIS

        self.initial_position = None
        self.armChain = None

        self._initial_position_used = None
        self._last_result = None
        self._psList = None
    
    def _initialize(self, initial_position, armChain, psList):
        self.initial_position = initial_position
        self.armChain = armChain
        self._psList = psList
    
    def _is_initialized(self):
        return self.armChain != None and self._psList != None

    def step(self, **kwargs):
        """
            @return 回傳各關節的目標角度
        """
        # 如果起始姿勢沒變，就沿用舊結果
        if (self.lazyIK and type(self.initial_position) != type(None) and (self.initial_position == self._initial_position_used).all()):
            return self._last_result

        # 否則起始姿勢有變、第一次呼叫，才計算IK
        if (self.orientation == None):
            ikResults = self.armChain.inverse_kinematics(
                target_position=self.target_position, 
                initial_position=self.initial_position)
        else:
            ikResults = self.armChain.inverse_kinematics(
                target_position=self.target_position, 
                #target_orientation=[0,0,-1],
                target_orientation=self.orientation,
                orientation_mode=self.AXIS,
                initial_position=self.initial_position)

        self._last_result = ikResults[1:-2]
        self._initial_position_used = self.initial_position
        
        return self._last_result
        
    def done(self):
        psValue = PSFunc.getInitialPosition(self._psList)
        fk = self.armChain.forward_kinematics(psValue)
        fingertipPosition = fk[:3, 3]
        distance = np.linalg.norm(fingertipPosition - np.array(self.target_position))
        if (self.AXIS == "X"):
            fingerOrientation = fk[0:3, 0]
        elif (self.AXIS == "Y"):
            fingerOrientation = fk[0:3, 1]
        elif (self.AXIS == "Z"):
            fingerOrientation = fk[0:3, 2]
            
        if (self.orientation != None):
            cos = np.dot(fingerOrientation, self.orientation)
        else:
            cos = 1
        
        #print(fk)
        #print(distance, 1-abs(cos), fingerOrientation, self.orientation)
        #print(distance, cos)
        print(distance <= self.tolerance , 1-abs(cos) < self.o_tolerance, self.o_tolerance)
        if (distance <= self.tolerance and 1-abs(cos) < self.o_tolerance):
            return True
        return False
        
class Grab(Action):
    def __init__(self, isGrab=True):
        self.isGrab = isGrab
        self.psList = []
        self.doneCounter = 5
    
    def _is_initialized(self):
        return (self.psList != [])
        
    def _initialize(self, psList):
        self.psList = PSFunc.getValue(psList)
        if(self.isGrab):
            self.psList.extend([0.00, 0.00])
        else:
            self.psList.extend([0.04, 0.04])
        
    def step(self):
        self.doneCounter -= 1
        return np.array(self.psList)

    def done(self):
        return (self.doneCounter <= 0)
        
class Turn(Action):
    def __init__(self, angle):
        self.direction_list = [angle,0,0,0,0,0,0]
        self.step_distance_list = [abs(-2.8973 - 2.8973),0,0,0,0,0,0]
        self.actuate = Actuate(step_distance_list = self.direction_list, direction_list = self.step_distance_list)
    
    def _is_initialized(self):
        return self.actuate._is_initialized()
        
    def _initialize(self, psList):
        self.actuate._initialize(psList)
        
    def step(self):
        return self.actuate.step()

    def done(self):
        return self.actuate.done()
        
class Pour(Action):
    def __init__(self, angle):
        self.direction_list = [0,0,0,0,0,0,angle]
        self.step_distance_list = [0,0,0,0,0,0, abs(-2.8973 - 2.8973)]
        self.actuate = Actuate(step_distance_list = self.direction_list, direction_list = self.step_distance_list)
    
    def _is_initialized(self):
        return self.actuate._is_initialized()
        
    def _initialize(self, psList):
        self.actuate._initialize(psList)
        
    def step(self):
        return self.actuate.step()

    def done(self):
        return self.actuate.done()

class Script:
    """
        負責管控腳本進度
    """
    def __init__(self, actions):
        """
            @param
                actions: Action陣列；此腳本要執行的動作
        """
        if (type(actions) != type([]) or len(actions) <= 0):
            raise TypeError("'actions' must be a list of Actions")
        self.actions = actions
        self.action_done = [False for _ in range(len(actions))]
        self.current_action_id = 0
        self._armChain = None
        self._psList = None

    ##### 要禁止 已經開始之後才加新的動作嗎
    def append(self, action):
        """
            向後塞入一個動作(Action)
        """
        self.actions.append(action)
        self.actions_done.append(False)

    def _initialize(self, armChain, psList):
        self._armChain = armChain
        self._psList = psList

    #####
    def step(self, **kwargs):
        # #####
        # 判斷此步驟是否執行完成 & 找到下一個要執行的動作
        # #####
        while True:
            if (self.current_action_id >= len(self.actions)):
                return None
            
            ##### TODO:  也許每次都對下一個action做初始化，這樣就可以重複利用action
            if (self.actions[self.current_action_id]._is_initialized() == False):
                self._initialize_action()

            if (self.actions[self.current_action_id].done()):
                self.action_done[self.current_action_id] = True
                self.current_action_id += 1
                print("Action done.")
            else:
                break

        return self.actions[self.current_action_id].step(**kwargs)

    def _initialize_action(self):
        current_action = self.actions[self.current_action_id]
        if (IKReachTarget == type(current_action)):
            initPos = PSFunc.getInitialPosition(self._psList)
            current_action._initialize(initPos, self._armChain, self._psList)

        elif (Actuate == type(current_action)):
            current_action._initialize(self._psList)

        elif (Pause == type(current_action)):
            pass
            
        elif (Grab == type(current_action)):
            current_action._initialize(self._psList)
            
        elif (Turn == type(current_action)):
            current_action._initialize(self._psList)
            
        elif (Pour == type(current_action)):
            current_action._initialize(self._psList)

        else:
            print()
            raise TypeError("Trying to initialize unknown action in script")

    #####
    def done(self):
        return self.current_action_id >= len(self.actions)
        # return self.action_done[self.current_action_id]