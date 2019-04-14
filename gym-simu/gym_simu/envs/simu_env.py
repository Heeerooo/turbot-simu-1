import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np

from Simulator import Simulator
from FakeSpeedController import SpeedController
from FakeVoiture import Voiture

class SimuEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        ###########################
        # Create and init simulator
        ###########################

        simulator = Simulator()
        self.simulator = simulator

        right_motor = simulator.get_handle("driving_joint_rear_right")
        left_motor = simulator.get_handle("driving_joint_rear_left")
        left_steering = simulator.get_handle("steering_joint_fl")
        right_steering = simulator.get_handle("steering_joint_fr")
        self.gyro = "gyroZ"
        self.cam = simulator.get_handle("Vision_sensor")
        self.base_car = simulator.get_handle("base_link")

        speedController = SpeedController(simulator, [left_motor, right_motor],
                                        simulation_step_time=simulator.get_simulation_time_step(),
                                        base_car=self.base_car)
        self.voiture = Voiture(simulator, [left_steering, right_steering], [left_motor, right_motor],
                        speedController, )

        self.simulator.start_simulation()

        ###############################
        # Create and init actions space
        ###############################

        min_steering = -100.0
        max_steering = 100.0
        min_speed = -100.0
        max_speed = 100.0

        min_action = np.array([min_steering, min_speed])
        max_action = np.array([max_steering, max_speed])

        self.viewer = None

        self.action_space = spaces.Box(low=min_action, high=max_action,
                                dtype=np.float32)

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

        # Take action
        self.voiture.tourne(action[0])
        self.voiture.avance(action[1])
        print("Applying control. Steering: ", action[0], " Speed: ", action[1])

        print("Entering simulator step")

        # Execute simulation
        self.simulator.do_simulation_step()

        print("Exit simulator step")

        reward = 0
        ob = None
        episode_over = False
        return ob, reward, episode_over, {}

    def reset(self):
        self.simulator.reset_simulation()

    def close(self):
        self.simulator.stop_simulation()

    def render(self, mode='human', close=False):
        pass

    def get_reward(self):
        """ Reward is given for XY. """
        if self.status == FOOBAR:
            return 1
        elif self.status == ABC:
            return self.somestate ** 2
        else:
            return 0