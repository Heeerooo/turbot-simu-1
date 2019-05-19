import time

import gym
import numpy as np
from gym import spaces

from robot.Car import Car
from robot.Gyro import Gyro
from robot.ImageAnalyzer import ImageAnalyzer
from robot.ImageEncoder import ImageEncoder
from robot.RewardManager import RewardManager
from robot.Simulator import Simulator
from robot.SpeedController import SpeedController
from robot.Tachometer import Tachometer


class SimuEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
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

        self.reward_manager = RewardManager()
        self.simulator.start_simulation()

        ###############################
        # Create actions space
        ###############################
        self.collision_distance = 0.05
        self.min_steering = -50.0   # TODO calibrate according to real car
        self.max_steering = 50.0
        self.min_speed = 20.0
        self.max_speed = 100.0
        self.steering = 0.0
        self.speed = 20.0

        self.resets_count = 0

        ##############################
        # Actions
        ##############################
        def nothing():
            pass

        def accelerate(value):
            self.speed += value

        def decelerate(value):
            self.speed -= value

        def turn(value):
            self.steering += value

        self.actions = {
            0: (nothing, None),
            1: (accelerate, 5),
            2: (decelerate, 5),
            3: (turn, 1),
            4: (turn, 10),
            5: (turn, -1),
            6: (turn, -10),
        }
        self.action_space = spaces.Discrete(len(self.actions))

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

        # execution times
        self.times = []
        self.rl_times = []
        self.last_step_end_time = None
        self.print_execution_times = False

        # Observations are encoded that way:
        # Channel 0: gyro
        # Channel 1: tacho
        # Channel 2: delta gyro
        # Channel 3: delta tacho
        # Channel 4: target steering
        # Channel 5: target speed
        # Channel 6..: image encoded features
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(6 + self.nb_features,), dtype='float32')

    def step(self, action_id):
        ##############################
        # Send action to simulator
        ##############################
        assert self.action_space.contains(action_id), "%r (%s) invalid" % (action_id, type(action_id))
        action_function, args = self.actions[action_id]
        action_function(args) if args is not None else action_function()

        self.steering = np.clip(self.steering, self.min_steering, self.max_steering)
        self.speed = np.clip(self.speed, self.min_speed, self.max_speed)
        self.car.tourne(self.steering)
        self.car.avance(self.speed)

        # Robot controls
        self.speed_controller.execute()

        if self.last_step_end_time is not None:
            self.rl_times.append(time.time() - self.last_step_end_time)
        # Execute simulation
        start_time = time.time()
        self.simulator.do_simulation_step()
        self.times.append(time.time() - start_time)
        self.last_step_end_time = time.time()

        # Robot controls
        components = [self.gyro, self.tachometer, self.image_analyzer, self.image_encoder]
        for component in components:
            component.execute()

        ####################################
        # Get rendered values from simulator
        ####################################
        return self.get_obs(), self.get_reward(), self.check_collision_with_wall(), {}

    def reset(self):
        # Reset simulation
        self.simulator.teleport_to_target("virage")
        self.simulator.do_simulation_step()
        self.steering = 0.0
        self.speed = 0.0
        self.gyro.reset()
        self.tachometer.reset()
        self.reward_manager.reset()
        if self.print_execution_times:
            print("Mean simulator execution Time : %f" % np.mean(self.times))
            print("Mean rl execution Time : %f" % np.mean(self.rl_times))
        self.times = []
        self.resets_count += 1
        return self.get_obs()

    def close(self):
        self.simulator.stop_simulation()

    def render(self, mode='human', close=False):
        pass

    def get_obs(self):
        # Put observations in a tensor
        return np.array([self.get_gyro(),
                         self.get_tacho(),
                         self.gyro.get_delta_cap(),
                         self.tachometer.get_delta_tacho(),
                         self.steering,
                         self.speed,
                         *self.image_encoder.get_encoded_image()])

    def get_reward(self):

        current_pos = self.simulator.get_object_position(self.handles['base_car'])

        reward = self.reward_manager.get_reward(current_pos) if current_pos is not None else 0.0

        # Add a reward for keeping high distance to walls
        reward += self.get_distance_with_walls() - 0.62

        # Add a constant penalty for each step (to minimize number of steps)
        reward -= 0.001

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

    def check_collision_with_wall(self):
        return self.get_distance_with_walls() < self.collision_distance
