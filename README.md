# Pharmaceutical-Robotic-Arm
109 NTHU CS Undergraduate Research Project Competition

## Version with only IKPY
This branch contains a version that uses only inverse kinematics to perform the action of reaching the beaker, lifting it up, move it to the target beaker, pour out the content, return it to its original position, then move the arm away.
The main contribution of this branch is that it sets the foundation on which we can readily replace certain actions with RL-based controller, i.e., the `PandaController.py` module.
### Demo
![image](https://github.com/CT-Lab/Pharmaceutical-Robotic-Arm/blob/IKPY_Only/IK_demo.gif)

* `PandaController.py`
  * A little background knowledge first.  In webots, we controll robots with its `motor`'s, by telling `motor` to turn to an angle or with a certain speed; we can get the information of joints with `positionSensor`, i.e., at what angle is it turned; but we cannot acquire the position of where each joint is in 3D space.
The way webots works is that, at each simulation step, we recieve information from the environment, and we are allowed to manipulate our robot.

* `PandaController`
  * The `PandaController` class is the unit that directly controlls our robotic arm.  It is responsible for moving each joint by performing whatever action `Script` decides.

* `Script`
  * The `Script` class is as its name suggests.  `Script` takes in the specified `Action`'s and performing them one by one, while providing whatever data they need.

* `Action`
  * `Action` is a skeleton that defines the format `Script` accepts that a subclass needs to follow.
Examples of `Action` include `Actuate`, `IKReachTarget`, `Pause`.
