import gym
import numpy as np
from gym import spaces

from robot.Car import Car
from robot.Gyro import Gyro
from robot.ImageAnalyzer import ImageAnalyzer
from robot.ImageEncoder import ImageEncoder
from robot.Simulator import Simulator
from robot.SpeedController import SpeedController
from robot.Tachometer import Tachometer


class SimuEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.speed_controller = None
        self.image_analyzer = None
        self.tachometer = None
        self.gyro = None
        self.car = None

        # TODO remove these lines after creating real reward
        self.old_tacho_value = 0
        self.delta_tacho = 0
        self.last_current_pos = None

        ###########################
        # Create and init simulator
        ###########################
        self.simulator = Simulator()

        self.handles = {
            "right_motor": self.simulator.get_handle("driving_joint_rear_right"),
            "left_motor": self.simulator.get_handle("driving_joint_rear_left"),
            "left_steering": self.simulator.get_handle("steering_joint_fl"),
            "right_steering": self.simulator.get_handle("steering_joint_fr"),
            "cam": self.simulator.get_handle("Vision_sensor"),
            "base_car": self.simulator.get_handle("base_link"),
            "int_wall": self.simulator.get_handle("int_wall"),
            "ext_wall": self.simulator.get_handle("ext_wall"),
            "body_chasis": self.simulator.get_handle("body_chasis")
        }
        self.gyro_name = "gyroZ"

        # Create robot control objects
        self._recreate_components()

        self.simulator.start_simulation()

        ###############################
        # Create actions space
        ###############################

        self.min_steering = -100.0
        self.max_steering = 100.0
        self.min_speed = 0
        self.max_speed = 100.0
        self.coeff_action_steering = 1.0  # Multiplier (to adapt order of magnitude of actions)
        self.coeff_action_speed = 2.0  # Multiplier (to adapt order of magnitude of actions)
        self.center_speed = 50.0  # Medium speed

        min_action = np.array([self.min_steering * self.coeff_action_steering,
                               (self.min_speed - self.center_speed) * self.coeff_action_speed])
        max_action = np.array([self.max_steering * self.coeff_action_steering,
                               (self.max_speed - self.center_speed) * self.coeff_action_speed])

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

        self.nb_features = self.image_encoder.get_nb_features_encoding()  # Nb of features in the output of encoder

        # Observations are encoded that way:
        # Channel 0: image
        # Channel 1: all pixels = (gyro - min_gyro) * (max_gyro - min_gyro) * 255
        # Channel 2: all pixels = (tacho - min_tacho) * (max_tacho - min_tacho) * 255
        self.observation_space = spaces.Box(low=-50.0, high=50.0, shape=(2 + self.nb_features,), dtype='float32')

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

        steering = action[0] / self.coeff_action_steering
        speed = (action[1] + self.center_speed) / self.coeff_action_speed
        clipped_steering = np.clip(steering, self.min_steering, self.max_steering)
        clipped_speed = np.clip(speed, self.min_speed, self.max_speed)
        self.car.tourne(clipped_steering)
        self.car.avance(clipped_speed)
        print("\nApplying control. Steering: ", clipped_steering, " Speed: ", clipped_speed)
        print("Initial control. Steering: ", steering, " Speed: ", speed)

        # Compute penalty if actions are out of action space
        self.action_penalty = 0.0
        action_penalty_coeff = 20000
        self.action_penalty -= max(steering - self.max_steering, 0) / action_penalty_coeff
        self.action_penalty -= max(-steering + self.min_steering, 0) / action_penalty_coeff
        self.action_penalty -= max(speed - self.max_speed, 0) / action_penalty_coeff
        self.action_penalty -= max(-speed + self.min_speed, 0) / action_penalty_coeff

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

        # Penalize the reward when hitting a wall
        if episode_over:
            reward -= 100.0

        # print ("Exiting step")

        return ob, reward, episode_over, {}

    def reset(self):

        # Reset simulation
        self.simulator.teleport_to_start_pos()

        # Create robot control objects
        # self._recreate_components()

        obs = self._get_obs()
        return obs

    def close(self):
        self.simulator.stop_simulation()

    def render(self, mode='human', close=False):
        pass

    def get_reward(self):
        """ Reward is given for XY. """
        # TODO create real reward
        # return self.delta_tacho - 1.0
        current_pos = self.simulator.get_object_position(self.handles['base_car'])
        if current_pos is None:
            return 0.0
        current_pos = np.array(current_pos)[1]
        if self.last_current_pos is None:
            self.last_current_pos = current_pos
            return 0.0
        reward = current_pos - self.last_current_pos
        self.last_current_pos = current_pos
        # Add a constant penalty for each step (to minimize number of steps)
        reward -= 0.1
        # Add the action penalty if actions are out of action space
        reward += self.action_penalty
        reward *= 100  # Change order of magnitude of reward
        print("Reward: ", reward, "Action penalty: ", self.action_penalty)
        return reward

    def _get_obs(self):
        # Execute arduino and speedController
        components = [self.gyro, self.tachometer, self.image_analyzer, self.image_encoder, self.speed_controller]
        for component in components:
            component.execute()

        gyro_value = self.get_gyro()

        tacho_value = self.get_tacho()

        # TODO remove this and replace by real reward
        self.delta_tacho = tacho_value - self.old_tacho_value
        self.old_tacho_value = tacho_value

        # Get camera image
        image_line_encoded = self.image_encoder.get_encoded_image()

        # Put observations in a tensor
        ob = np.zeros((2 + self.nb_features))
        ob[0] = gyro_value
        ob[1] = tacho_value
        ob[2:] = image_line_encoded

        return ob

    def get_gyro(self):
        gyro_value = self.gyro.get_cap()
        return (gyro_value - self.min_gyro) / (self.max_gyro - self.min_gyro) * 100 - 50

    def get_tacho(self):
        tacho_value = self.tachometer.get_tacho()
        return tacho_value * (tacho_value - self.min_tacho) / (self.max_tacho - self.min_tacho) * 100 - 50

    def _recreate_components(self):

        self.speed_controller = SpeedController(simulator=self.simulator,
                                                motor_handles=[self.handles["left_motor"], self.handles["right_motor"]],
                                                simulation_step_time=self.simulator.get_simulation_time_step())

        self.image_analyzer = ImageAnalyzer(simulator=self.simulator,
                                            cam_handle=self.handles["cam"])

        self.image_encoder = ImageEncoder(image_analyzer=self.image_analyzer)

        self.tachometer = Tachometer(simulator=self.simulator,
                                     base_car=self.handles['base_car'])

        self.gyro = Gyro(simulator=self.simulator,
                         gyro_name=self.gyro_name)

        self.car = Car(simulator=self.simulator,
                       steering_handles=[self.handles["left_steering"], self.handles["right_steering"]],
                       motors_handles=[self.handles["left_motor"], self.handles["right_motor"]],
                       speed_controller=self.speed_controller,
                       tachometer=self.tachometer,
                       gyro=self.gyro)

    def _check_collision_with_wall(self):
        # TODO replace body_chassis by base_car ?

        # simxCheckCollision not working (crash when collision on NULL pointer)
        # success1, collision1 = self.simulator.client.simxCheckCollision(self.int_wall, self.body_chassis, self.simulator.client.simxServiceCall())
        # success2, collision2 = self.simulator.client.simxCheckCollision(self.ext_wall, self.body_chassis, self.simulator.client.simxServiceCall())

        # Added try/catch because sometimes it crashes. TODO: understand why
        try:
            list1 = self.simulator.client.simxCheckDistance(self.handles["int_wall"], self.handles["body_chasis"], 0.05,
                                                            self.simulator.client.simxServiceCall())
            list2 = self.simulator.client.simxCheckDistance(self.handles["ext_wall"], self.handles["body_chasis"], 0.05,
                                                            self.simulator.client.simxServiceCall())
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
