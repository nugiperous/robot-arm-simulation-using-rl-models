import gym
from gym import spaces
import pybullet as p
import pybullet_data
import numpy as np

class RobotArmEnv(gym.Env):
    def __init__(self):
        # Initialize PyBullet simulation
        p.connect(p.GUI)
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])
        #self.action_space = spaces.Box(np.array([-1]*4), np.array([1]*4))
        #self.observation_space = spaces.Box(np.array([-1]*5), np.array([1]*5))
        
        
    def define_goal(self, target_block:int):
        # Setup of Goal
        if target_block == 1:
            self.target_block = self.jengaUid1
        elif target_block == 2:
            self.target_block = self.jengaUid2
        elif target_block == 3:
            self.target_block = self.jengaUid3            


    def reset(self):
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
        
        # Load robot arm model and get joint information
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        planeID = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
        self.num_joints = p.getNumJoints(self.robot_id)
        
        # Reset robot arm to initial state
        for i in range(self.num_joints):
            p.resetJointState(self.robot_id, i, targetValue=0.0, targetVelocity=0.0)
        
        # Define observation space
        obs_low = np.array([-np.pi] * self.num_joints)  # lower bound of joint angles
        obs_high = np.array([np.pi] * self.num_joints)  # upper bound of joint angles
        self.observation_space = spaces.Box(low=obs_low, high=obs_high)
        
        # Define action space
        act_low = np.array([-1.0] * self.num_joints)  # lower bound of joint torques
        act_high = np.array([1.0] * self.num_joints)  # upper bound of joint torques
        self.action_space = spaces.Box(low=act_low, high=act_high)

        state_robot = p.getLinkState(self.robot_id, 11)[0]
        state_fingers = (p.getJointState(self.robot_id,9)[0], p.getJointState(self.robot_id, 10)[0])
        observation = state_robot + state_fingers
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)

        # Set Gravity
        p.setGravity(0,0,-9.81)
        # Set Realtime Simulation
        p.setRealTimeSimulation(1)
        
        # Load Jenga Blocks
        self.jengaUid1 = p.loadURDF("jenga/jenga.urdf", basePosition=[.7,-0.25,.25], baseOrientation = [0,1,0,1])
        self.jengaUid1_org_pos, orn = p.getBasePositionAndOrientation(self.jengaUid1)
        self.jengaUid2 = p.loadURDF("jenga/jenga.urdf", basePosition=[.7,0,.25], baseOrientation = [0,1,0,1])
        self.jengaUid2_org_pos, orn = p.getBasePositionAndOrientation(self.jengaUid2)
        self.jengaUid3 = p.loadURDF("jenga/jenga.urdf", basePosition=[.7,0.25,.25], baseOrientation = [0,1,0,1])
        self.jengaUid3_org_pos, orn = p.getBasePositionAndOrientation(self.jengaUid3)
        
        self.tableUid1 = p.loadURDF("table/table.urdf", basePosition=[1,0,-0.5])

        return np.zeros(self.num_joints)

    def reset_(self):
        # Reset robot arm to initial state
        p.resetSimulation()
        #p.resetJointState(self.robot_id, range(self.num_joints), [0.0] * self.num_joints)
        for i in range(self.num_joints):
            p.resetJointState(self.robot_id, i, targetValue=0.0, targetVelocity=0.0)
        
        # Return initial observation
        return np.zeros(self.num_joints)

    def step(self, action):
        # Apply joint torques to the robot arm
        joint_positions = action
        joint_velocities = [0.0]*self.num_joints
        p.setJointMotorControlArray(self.robot_id, range(self.num_joints), p.POSITION_CONTROL, targetPositions=joint_positions, targetVelocities=joint_velocities)
        # Step the simulation
        p.stepSimulation()
        # Get current joint angles as observation
        observation = []
        for i in range(self.num_joints):
            joint_info = p.getJointState(self.robot_id, i)
            joint_angle = joint_info[0]
            observation.append(joint_angle)
        observation = np.array(observation)
        # Compute reward, done, and info
        reward = self.compute_reward(self.target_block)  
        
        # Termination Condition
        done = self.compute_done(self.target_block)
        info = {}  # additional information
        return observation, reward, done, info

    def compute_reward(self, blockID):
        # Find Jenga Block 1 displacement 
        block1_pos, orn = p.getBasePositionAndOrientation(self.jengaUid1)
        block1_displacement = np.linalg.norm(np.array(self.jengaUid1_org_pos) - np.array(block1_pos))
        
        # Find Jenga Block 3 displacement 
        block3_pos, orn = p.getBasePositionAndOrientation(self.jengaUid3)
        block3_displacement = np.linalg.norm(np.array(self.jengaUid3_org_pos) - np.array(block3_pos))
        
        # Get current position of the targeted jenga block
        pos, orn = p.getBasePositionAndOrientation(blockID) # pos = x,y,z ; orn = w,x,y,z
        
        
        # Get positions of the gripper links (links 9 and 10)
        gripper_tip_positions = [p.getLinkState(self.robot_id, i)[0] for i in [9, 10]]
        
        # Calculate distance between each gripper tip and the object
        distances = [np.linalg.norm(np.array(gripper_tip_pos) - np.array(pos)) for gripper_tip_pos in gripper_tip_positions]
        gripper_to_block = min(distances)
        
        reward = -gripper_to_block - block1_displacement - block3_displacement

        return reward
        
    # def compute_done(self, blockID):
    #     pos, orn = p.getBasePositionAndOrientation(blockID)
    #     if np.array_equal(pos, self.target_block_pos):
    #         return True
    #     else:
    #         return False
        
    def compute_done(self, blockID):
        block_pos, orn = p.getBasePositionAndOrientation(blockID)
        gripper_tip_positions = [p.getLinkState(self.robot_id, i)[0] for i in [9, 10]]
        dist = [np.linalg.norm(np.array(gripper_tip_pos) - np.array(block_pos)) for gripper_tip_pos in gripper_tip_positions]
        gripper_to_block = min(dist)
        #dist = np.linalg.norm(np.array(pos) - np.array(self.target_block_pos))
        
        org_block_dist = np.linalg.norm(np.array(self.jengaUid2_org_pos) - np.array(block_pos))
        
        #print(gripper_to_block)
        if gripper_to_block < .1:
            #print("dist = ", gripper_to_block)
            print("DONE")
            return True
        elif org_block_dist > 1:
            return True
        else:
            return False
        
    def render(self, mode='human'):
        if mode == 'human':
            # Render in graphical mode
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        else:
            # Render in non-graphical mode (e.g., for headless servers)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    
        # Set camera parameters
        camera_distance = 1.5
        camera_pitch = -40
        camera_yaw = 0.0
        camera_target_position = [0.55,-0.35,0.2]
        p.resetDebugVisualizerCamera(camera_distance, camera_pitch, camera_yaw, camera_target_position)
    
        # Update the visualization
        p.removeAllUserDebugItems()
        p.stepSimulation()
    
        return p.getCameraImage(640, 480)  # Return the rendered image or other visualization data

    def close(self):
        # Close the environment and PyBullet simulation
        p.disconnect()


