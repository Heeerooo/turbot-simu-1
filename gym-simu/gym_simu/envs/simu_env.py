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
        self.last_reward = 0.0

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

        self.simulator.start_simulation()

        ###############################
        # Create actions space
        ###############################

        self.min_steering = -100.0
        self.max_steering = 100.0
        self.min_speed = 20.0
        self.max_speed = 100.0
        self.coeff_action_steering = 1.0  # Multiplier (to adapt order of magnitude of actions)
        self.coeff_action_speed = 2.0  # Multiplier (to adapt order of magnitude of actions)
        self.center_speed = 50.0  # Medium speed
        self.steering = 0.0
        self.speed = 0.0

        self.viewer = None

        # Action space
        # 0: do nothing
        # 1: +5 speed
        # 2: -5 speed
        # 3: +1 steering
        # 4: +10 steering
        # 5: -1 steering
        # 6: -10 steering
        self.action_space = spaces.Discrete(7)

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
        # Channel 0: gyro
        # Channel 1: tacho
        # Channel 2: target steering
        # Channel 3: target speed
        # Channel 4..: image encoded features
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(4 + self.nb_features,), dtype='float32')

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
        ##############################
        # Send action to simulator
        ##############################

        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        if action == 1:
            self.speed += 5
        elif action == 2:
            self.speed -= 5
        elif action == 3:
            self.steering += 1
        elif action == 4:
            self.steering += 10
        elif action == 5:
            self.steering -= 1
        elif action == 6:
            self.steering -= 10
        self.steering = np.clip(self.steering, self.min_steering, self.max_steering)
        self.speed = np.clip(self.speed, self.min_speed, self.max_speed)
        self.car.tourne(self.steering)
        self.car.avance(self.speed)

        # Robot controls
        components = [self.gyro, self.tachometer, self.image_analyzer, self.image_encoder, self.speed_controller]
        for component in components:
            component.execute()

        # Execute simulation
        self.simulator.do_simulation_step()

        ####################################
        # Get rendered values from simulator
        ####################################

        ob = self.get_obs()

        distance_to_walls = self.get_distance_with_walls()

        reward = self.get_reward(distance_to_walls)

        episode_over = self.check_collision_with_wall(distance_to_walls)

        return ob, reward, episode_over, {}

    def reset(self):
        # Reset simulation
        self.simulator.teleport_to_start_pos()
        self.simulator.do_simulation_step()
        self.steering = 0.0
        self.speed = 0.0
        self.gyro.reset()
        self.tachometer.reset()
        self.last_current_pos = None

        return self.get_obs()

    def close(self):
        self.simulator.stop_simulation()

    def render(self, mode='human', close=False):
        pass

    def get_obs(self):

        gyro_value = self.get_gyro()

        tacho_value = self.get_tacho()

        # TODO remove this and replace by real reward
        self.delta_tacho = tacho_value - self.old_tacho_value
        self.old_tacho_value = tacho_value

        # Get camera image
        image_line_encoded = self.image_encoder.get_encoded_image()

        # Put observations in a tensor
        ob = np.zeros((4 + self.nb_features))
        ob[0] = gyro_value
        ob[1] = tacho_value
        ob[2] = self.steering
        ob[3] = self.speed
        ob[4:] = image_line_encoded

        return ob

    def get_reward(self, distance_to_walls):
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

        # Add a reward for keeping high distance to walls
        reward += distance_to_walls - 0.62

        # Add a constant penalty for each step (to minimize number of steps)
        reward -= 0.001

        print("\n")
        print("Step reward: %f " % reward)

        return reward

    def get_gyro(self):
        gyro_value = self.gyro.get_cap()
        return (gyro_value - self.min_gyro) / (self.max_gyro - self.min_gyro) * 100 - 50

    def get_tacho(self):
        tacho_value = self.tachometer.get_tacho()
        return tacho_value * (tacho_value - self.min_tacho) / (self.max_tacho - self.min_tacho) * 100 - 50

    def get_distance_with_walls(self):
        road_width = 1.5
        distance_int = self.simulator.get_distance(self.handles["body_chasis"], self.handles["int_wall"],
                                                   road_width / 2)
        distance_ext = self.simulator.get_distance(self.handles["body_chasis"], self.handles["ext_wall"],
                                                   road_width / 2)
        return min(distance_int, distance_ext)

    def check_collision_with_wall(self, distance):
        return distance < 0.05
