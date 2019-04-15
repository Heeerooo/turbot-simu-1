import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np

from Simulator import Simulator
from FakeSpeedController import SpeedController
from FakeVoiture import Voiture
from FakeArduino import Arduino
from FakeImageAnalyser import ImageAnalyser

class SimuEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # TODO remove these lines after creating real reward
        self.old_tacho_value = 0
        self.delta_tacho = 0

        ###########################
        # Create and init simulator
        ###########################

        simulator = Simulator()
        self.simulator = simulator

        # Create robot control objects
        self._create_robot_control_objects(self.simulator)

        # Execute arduino and speedController a first time
        components = [self.arduino, self.imageAnalyser, self.speedController]
        for component in components:
            component.execute()

        self.simulator.start_simulation()

        ###############################
        # Create actions space
        ###############################

        self.min_steering = -100.0
        self.max_steering = 100.0
        self.min_speed = 0.0
        self.max_speed = 100.0

        min_action = np.array([self.min_steering, self.min_speed])
        max_action = np.array([self.max_steering, self.max_speed])

        self.viewer = None

        self.action_space = spaces.Box(low=min_action, high=max_action,
                                dtype=np.float32)

        ##############################
        # Create observations space
        ##############################

        self.min_gyro = -180.0
        self.max_gyro = 180.0
        self.min_tacho = -50000.0
        self.max_tacho = 50000.0
        self.width = 320
        self.height = 240

        # Observations are encoded that way:
        # Channel 0: image
        # Channel 1: all pixels = (gyro - min_gyro) * (max_gyro - min_gyro) * 255
        # Channel 2: all pixels = (tacho - min_tacho) * (max_tacho - min_tacho) * 255
        self.observation_space = spaces.Box(low=0.0, high=255.0, shape=(self.height, self.width, 3), dtype='float32')

    def step(self, action):
        """

        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        # self._take_action(action)
        # self.status = self.env.step()
        # reward = self.get_reward()
        # ob = self.env.getState()
        # episode_over = self.status != hfo_py.IN_GAME

        ##############################
        # Send action to simulator
        ##############################

        self.voiture.tourne(np.clip(action[0], self.min_steering, self.max_steering))
        self.voiture.avance(np.clip(action[1], self.min_speed, self.max_speed))
        print("Applying control. Steering: ", action[0], " Speed: ", action[1])

        # print("Entering simulator step")

        # Execute simulation
        self.simulator.do_simulation_step()

        # print("Exit simulator step")

        ####################################
        # Get rendered values from simulator
        ####################################

        ob = self._get_obs()

        reward = self.get_reward()

        episode_over = self._check_collision_with_wall()

        # print ("Exiting step")

        return ob, reward, episode_over, {}

    def reset(self):
        # Reset simulation
        self.simulator.reset_simulation()

        # Reset robot control objects
        self._create_robot_control_objects(self.simulator)

        obs = self._get_obs()
        return obs

    def close(self):
        self.simulator.stop_simulation()

    def render(self, mode='human', close=False):
        pass

    def get_reward(self):
        """ Reward is given for XY. """
        # TODO create real reward
        return self.delta_tacho

    def _get_obs(self):
        # Execute arduino and speedController
        components = [self.arduino, self.imageAnalyser, self.speedController]
        for component in components:
            component.execute()

        # Get gyro
        gyro_value = self.arduino.gyro
        gyro_matrix = np.ones((self.height, self.width), dtype='float32') * (gyro_value - self.min_gyro) / (self.max_gyro - self.min_gyro) * 255

        # Get tacho
        tacho_value = self.speedController.get_tacho()
        tacho_matrix = np.ones((self.height, self.width), dtype='float32') * (tacho_value - self.min_tacho) / (self.max_tacho - self.min_tacho) * 255

        # TODO remove this and replace by real reward
        self.delta_tacho = tacho_value - self.old_tacho_value
        self.old_tacho_value = tacho_value

        # Get camera image
        image_ligne = self.imageAnalyser.get_image_ligne()

        # Put observations in a tensor
        ob = np.zeros((self.height, self.width, 3))
        ob[..., 0] = image_ligne
        ob[..., 1] = gyro_matrix
        ob[..., 2] = tacho_matrix

        return ob

    def _create_robot_control_objects(self, simulator):
        right_motor = simulator.get_handle("driving_joint_rear_right")
        left_motor = simulator.get_handle("driving_joint_rear_left")
        left_steering = simulator.get_handle("steering_joint_fl")
        right_steering = simulator.get_handle("steering_joint_fr")
        gyro = "gyroZ"
        cam = simulator.get_handle("Vision_sensor")
        self.base_car = simulator.get_handle("base_link")
        self.int_wall = simulator.get_handle("int_wall")
        self.ext_wall = simulator.get_handle("ext_wall")
        self.body_chassis = simulator.get_handle("body_chasis")

        self.speedController = SpeedController(simulator, [left_motor, right_motor],
                                        simulation_step_time=simulator.get_simulation_time_step(),
                                        base_car=self.base_car)
        self.voiture = Voiture(simulator, [left_steering, right_steering], [left_motor, right_motor],
                        self.speedController, )

        self.imageAnalyser = ImageAnalyser(simulator, cam)

        self.arduino = Arduino(simulator, gyro)

    def _check_collision_with_wall(self):
        # TODO replace body_chassis by base_car ?

        # simxCheckCollision not working (crash when collision on NULL pointer)
        # success1, collision1 = self.simulator.client.simxCheckCollision(self.int_wall, self.body_chassis, self.simulator.client.simxServiceCall())
        # success2, collision2 = self.simulator.client.simxCheckCollision(self.ext_wall, self.body_chassis, self.simulator.client.simxServiceCall())
        
        try:
            list1 = self.simulator.client.simxCheckDistance(self.int_wall, self.body_chassis, 0.05,  self.simulator.client.simxServiceCall())
            list2 = self.simulator.client.simxCheckDistance(self.ext_wall, self.body_chassis, 0.05,  self.simulator.client.simxServiceCall())
        except:
            print("Warning: cannot compute distance to walls")
            return False

        # simxCheckDistance has a weird behaviour: it returns lists of len 2 if distance > threshold, len 5 otherwise.
        # Other issue: it returns success only if distance < threshold, otherwise other values are None.
        # Thus we handle this.
        if len(list1) == 2:
            success1, collision1 = list1
        elif len(list1) == 5:
            success1, collision1, _, _, _ = list1
        else:
            success1 = False

        if len(list2) == 2:
            success2, collision2 = list2
        elif len(list2) == 5:
            success2, collision2, _, _, _ = list2
        else:
            success2 = False

        # Get collision
        collision = False
        if success1:
            collision = collision or collision1
        if success2:
            collision = collision or collision2

        # print ("Collision: ", collision)

        return collision