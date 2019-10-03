# encoding:utf-8
import json

from InferenceEnable import disable_inference, enable_inference
from circle import angle_intersection
from robot.Component import Component

RAD_TO_DEG = 57.2958


class Sequencer(Component):
    # Durees d'appui sur le bouton poussoir
    DUREE_APPUI_COURT_REDEMARRAGE = 2  # Nombre de secondes d'appui sur le poussoir pour reinitialiser le programme
    DUREE_APPUI_LONG_SHUTDOWN = 10  # Nombre de secondes d'appui sur le poussoir pour eteindre le raspberry

    tacho = 0
    cap_target = 0.0
    sequence = 0
    start_sequence = True
    time_start = 0
    current_program = {}

    timer_led = 0
    vitesse_clignote_led = 10
    led_clignote = True
    last_led = 0

    timer_bouton = 0
    last_bouton = 1  # 1 = bouton relache, 0 = bouton appuye
    flag_appui_court = False  # Passe a True quand un appui court (3 secondes) a ete detecte

    strategy = None
    speed = None

    def __init__(self, car, program, strategy_factory, image_analyzer, start_light_detector):
        if start_light_detector is None:
            print("Warning no start light detector")
        self.start_light_detector = start_light_detector
        self.strategy_factory = strategy_factory
        self.car = car
        self.program = program
        self.image_analyzer = image_analyzer

    def execute(self):
        # Fait clignoter la led
        self.handle_led()

        if self.start_sequence:
            self.handle_start_sequence()

        if self.strategy is not None:
            steering = self.strategy.compute_steering()
            if steering is not None:
                self.car.turn(steering)
            speed = self.strategy.compute_speed()
            if speed is not None:
                self.car.forward(speed)

        if self.check_end_sequence():
            self.handle_end_sequence()

    def handle_led(self):
        if self.led_clignote:
            if self.car.get_time() > self.timer_led + self.vitesse_clignote_led:
                self.timer_led = self.car.get_time()
                self.last_led = False if self.last_led else True
                self.car.set_led(self.last_led)
        else:
            self.car.set_led(True)

    def set_tacho(self):
        self.tacho = self.car.get_tacho()

    def turn(self):
        steering = self.current_program['positionRoues']
        self.car.turn(steering)

    def forward(self):
        vitesse = self.current_program['speed']
        self.car.forward(vitesse)

    def passs(self):
        pass

    def init_lao(self):
        additional_offset = self.current_program['offset'] if 'offset' in self.current_program else 0
        angle_coef = self.current_program['angle_coef'] if 'angle_coef' in self.current_program else None
        offset_coef = self.current_program['offset_coef'] if 'offset_coef' in self.current_program else None
        self.strategy = self.strategy_factory.create_lao(additional_offset, angle_coef, offset_coef)

    def init_circle(self):
        p_coef = self.current_program['p_coef'] if 'p_coef' in self.current_program else 0
        i_coef = self.current_program['i_coef'] if 'i_coef' in self.current_program else 0
        d_coef = self.current_program['d_coef'] if 'd_coef' in self.current_program else 0
        obstacle_offset = self.current_program['obstacle_offset'] if 'obstacle_offset' in self.current_program else None
        avoidance_speed = self.current_program[
            'avoidance_speed'] if 'avoidance_speed' in self.current_program else self.speed
        self.strategy = self.strategy_factory.create_circle(p_coef,
                                                            i_coef,
                                                            d_coef,
                                                            self.speed,
                                                            avoidance_speed,
                                                            obstacle_offset)

    def init_cap_standard(self):
        cap_target = self.current_program[
            'cap_target'] if 'cap_target' in self.current_program else 0
        self.strategy = self.strategy_factory.create_cap_standard(cap_target, self.current_program['speed'])

    def init_cap_offset(self):
        speed = self.current_program['speed']
        p_correction_coef = self.current_program[
            'p_correction_coef'] if 'p_correction_coef' in self.current_program else 0
        i_correction_coef = self.current_program[
            'i_correction_coef'] if 'i_correction_coef' in self.current_program else 0
        cap_target = self.current_program[
            'cap_target'] if 'cap_target' in self.current_program else 0
        self.strategy = self.strategy_factory.create_cap_offset((cap_target + self.base_gyro) % 360, speed,
                                                                p_correction_coef,
                                                                i_correction_coef)

    def init_turn_offset(self):
        target_steering = self.current_program['steering_target']
        p_correction_coef = self.current_program[
            'p_correction_coef'] if 'p_correction_coef' in self.current_program else 0
        i_correction_coef = self.current_program[
            'i_correction_coef'] if 'i_correction_coef' in self.current_program else 0
        self.strategy = self.strategy_factory.create_turn_offset(target_steering, p_correction_coef, i_correction_coef)

    def handle_start_sequence(self):
        # Premiere execution de l'instruction courante
        self.current_program = self.program[self.sequence]
        instruction = self.current_program['instruction']
        print("********** Nouvelle instruction *********** ")
        print(print(json.dumps(self.current_program, indent=4)))
        self.time_start = self.car.get_time()
        self.strategy = None
        self.image_analyzer.reset()

        # Applique l'instruction
        instructions_actions = {
            'setTacho': self.set_tacho,
            'tourne': self.turn,
            'lineAngleOffset': self.init_lao,
            'circle': self.init_circle,
            'ligneDroite': self.init_cap_standard,
            'capOffset': self.init_cap_offset,
            'turnOffset': self.init_turn_offset,
        }

        self.set_speed()

        if instruction not in instructions_actions.keys():
            raise Exception("Instruction " + instruction + " does not exist")
        instructions_actions[instruction]()

        self.set_additional_params()

        self.start_sequence = False

    def set_speed(self):
        if 'speed' in self.current_program:
            speed = self.current_program['speed']
            self.car.forward(speed)
            self.speed = speed

    def set_additional_params(self):
        if 'offset_baseline_height' in self.current_program:
            self.image_analyzer.set_offset_baseline_height(self.current_program['offset_baseline_height'])
        if 'circle_radius' in self.current_program:
            self.image_analyzer.set_circle_radius(self.current_program['circle_radius'])
        if 'slow_zone_radius' in self.current_program:
            self.image_analyzer.set_slow_zone_radius(self.current_program['slow_zone_radius'])
        if 'display' in self.current_program:
            self.car.send_display(self.current_program['display'])
        if 'chenillard' in self.current_program:
            self.car.set_chenillard(self.current_program['chenillard'])
        if 'clip' in self.current_program:
            self.image_analyzer.set_clip_length(self.current_program['clip'])
        if 'obstacles' in self.current_program:
            self.image_analyzer.set_process_obstacle(self.current_program['obstacles'])
        if 'lock_zone_radius' in self.current_program:
            self.image_analyzer.set_lock_zone_radius(self.current_program['lock_zone_radius'])
        if 'avoidance_zone_radius' in self.current_program:
            self.image_analyzer.set_avoidance_zone_radius(self.current_program['avoidance_zone_radius'])
        if 'start_light_detector' in self.current_program:
            print("start detector")
            disable_inference()
            self.start_light_detector.start()

    def check_cap(self):
        final_cap_mini = self.current_program['capFinalMini']
        cap_final_maxi = self.current_program['capFinalMaxi']
        return self.check_delta_cap_reached(final_cap_mini, cap_final_maxi)

    def check_delay(self):
        return (self.car.get_time() - self.time_start) > self.current_program['duree']

    def check_tacho(self):
        return self.car.get_tacho() > (self.tacho + self.current_program['tacho'])

    def end_now(self):
        return True

    def check_button(self):
        self.vitesse_clignote_led = 0.3
        self.led_clignote = True
        button_value = self.car.get_push_button() == 0
        if button_value:
            self.led_clignote = False
        return button_value

    def check_gyro_stable(self):
        gyro_stable = self.car.check_gyro_stable()
        if gyro_stable:
            self.base_gyro = self.car.get_cap()
        return gyro_stable

    def check_circle(self):
        circle_angle_max = self.current_program['circle_angle_max']
        self.image_analyzer.analyze()
        poly_2_coefs = self.image_analyzer.poly_2_coefs
        if poly_2_coefs is None:
            return False

        error_angle = angle_intersection(poly_2_coefs[0], poly_2_coefs[1], poly_2_coefs[2],
                                         self.image_analyzer.circle_poly2_intersect_radius,
                                         self.image_analyzer.final_image_height,
                                         self.image_analyzer.final_image_width)
        error_angle_deg = error_angle * RAD_TO_DEG
        print("error angle", error_angle_deg)
        return error_angle_deg > circle_angle_max

    def check_start_light(self):
        if self.start_light_detector.detect_start_light():
            self.start_light_detector.stop()
            enable_inference()
            return True
        else:
            return False

    def check_end_sequence(self):
        # Recupere la condition de fin
        end_condition = self.current_program['conditionFin']

        # Verifie si la condition de fin est atteinte
        end_conditions_check = {
            'cap': self.check_cap,
            'duree': self.check_delay,
            'tacho': self.check_tacho,
            'immediat': self.end_now,
            'attendBouton': self.check_button,
            'attendreGyroStable': self.check_gyro_stable,
            'circle': self.check_circle,
            'waitStartLight': self.check_start_light
        }

        if end_condition not in end_conditions_check.keys():
            raise Exception("End condition " + end_condition + "does not exist")
        return end_conditions_check[end_condition]()

    def handle_end_sequence(self):
        # Si le champ nextLabel est defini, alors il faut chercher le prochain element par son label
        if 'nextLabel' in self.current_program:
            next_label = self.current_program['nextLabel']
            for i in range(len(self.program)):
                if 'label' in self.program[i]:
                    if self.program[i]['label'] == next_label:
                        # On a trouve la prochaine sequence
                        self.sequence = i
        else:
            # Si le champ nextLabel n'est pas defini, on passe simplement a l'element suivant
            self.sequence += 1
        self.start_sequence = True

    def check_delta_cap_reached(self, final_cap_min, final_cap_max):

        absolute_cap_mini = (self.base_gyro + final_cap_min) % 360
        absolute_cap_maxi = (self.base_gyro + final_cap_max) % 360

        gap_cap_mini = (((self.car.get_cap() - absolute_cap_mini) + 180) % 360) - 180
        gap_cap_maxi = (((self.car.get_cap() - absolute_cap_maxi) + 180) % 360) - 180

        turn_over = (gap_cap_mini > 0 and gap_cap_maxi < 0)

        if turn_over:
            print("--------------- Fin de virage ----------------")
            print("CapTarget : ", self.cap_target, "Cap : ", self.car.get_cap(), " Ecart cap mini : ", gap_cap_mini,
                  " Ecart cap maxi : ", gap_cap_maxi)
            print("----------------------------------------------")

        return turn_over
