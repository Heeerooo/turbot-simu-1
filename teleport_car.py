import random
import time

from robot.Simulator import Simulator

simulation_duration_seconds = 1.5

simulator = Simulator()

handles = {
    "right_motor": simulator.get_handle("driving_joint_rear_right"),
    "left_motor": simulator.get_handle("driving_joint_rear_left"),
    "left_steering": simulator.get_handle("steering_joint_fl"),
    "right_steering": simulator.get_handle("steering_joint_fr"),
    "right_suspension": simulator.get_handle("suspension_front_right"),
    "left_suspension": simulator.get_handle("suspension_front_left"),
    "cam": simulator.get_handle("Vision_sensor"),
    "base_car": simulator.get_handle("base_link"),
    "steering_axis": simulator.get_handle("steering_axis_fr"),
    "damper_left": simulator.get_handle("damper_front_left")
}

gyro_name = "gyroZ"

simulator.start_simulation()
for n in range(0, 50):
    steering = random.randint(-5, 5) / 100
    print("steering %f " % steering)
    speed = random.randint(-100, -20)
    print("speed %f " % steering)
    start_time = time.time()
    while time.time() - start_time < simulation_duration_seconds:
        start_step_time = time.time()
        print("code execution time : %fs " % (time.time() - start_step_time))
        print(simulator.client.simxGetJointForce(handles["damper_left"], simulator.client.simxServiceCall()))
        print(simulator.get_object_position(handles["steering_axis"]))

        simulator.set_target_pos(handles["right_steering"], steering)
        simulator.set_target_pos(handles["left_steering"], steering)
        simulator.set_target_speed(handles["left_motor"], speed)
        simulator.set_target_speed(handles["right_motor"], speed)
        start_simulator_step_time = time.time()
        simulator.do_simulation_step()
        print("simulator execution time : %fs " % (time.time() - start_simulator_step_time))
    simulator.teleport_to_start_pos()
simulator.stop_simulation()