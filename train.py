"""
Model of a train.

"""

import numpy as np
from scipy.constants import g
from math import fabs


class SystemState(object):
    """
    Class to store both train and brake state.
    """

    def __init__(self, train_state, brake_state):
        self.train_state = train_state
        self.brake_state = brake_state

    def __str__(self):
        train_state_str = str(self.train_state)
        brake_state_str = str(self.brake_state)
        return train_state_str + "\n" + brake_state_str


class TrainParameter(object):
    """
    Stores the parameters of a train.
    """

    def __init__(self, M=3.0e+6, pbar=4.0e6,
                 plowerbar=-1.0e6, vbar=6.56,
                 alpha=1.67e4, a=7.36e4, b=0.0, c=101):
        self.M = M
        self.pbar = pbar
        self.plowerbar = plowerbar
        self.vbar = vbar
        self.alpha = alpha
        self.a = a
        self.b = b
        self.c = c


class TrainState(object):
    """
    Stores the state of the train.
    """

    def __init__(self, initial_position=0, initial_speed=0,
                 initial_engine_power=0):
        self.xi = initial_position
        self.v = initial_speed
        self.p = initial_engine_power

    def deep_copy(self):
        new_state = TrainState()
        new_state.xi = self.xi
        new_state.v = self.v
        new_state.p = self.p
        return new_state

    def __str__(self):
        string_to_return = """Position: %.3g (m)\nSpeed: %.3g (m/s)
        Engine power %.3g (W)""" % (self.xi, self.v, self.p)
        return string_to_return


class Train(object):
    """
    Models a train. All variables are in SI.
    """

    def __init__(self, train_parameter, initial_state,
                 brake, train_path, dt=1):
        self.x0 = initial_state
        self.state = initial_state
        self.parameter = train_parameter
        self.brake = brake
        self.train_path = train_path
        self._dt = dt

    def _compute_force(self, speed=None, p=None):

        if (p is None):
            p = self.state.p
        if (speed is None):
            speed = self.state.v

        abs_speed = np.absolute(speed)
        if isinstance(speed, np.ndarray):
            force_N = np.zeros(abs_speed.shape)
            idx_higher = abs_speed > self.parameter.vbar
            idx_other = np.logical_not(idx_higher)
            force_N[idx_higher] = p / abs_speed[idx_higher]
            force_N[idx_other] = p / self.parameter.vbar
        else:
            if (abs_speed > self.parameter.vbar):
                force_N = p / abs_speed
            else:
                force_N = p / self.parameter.vbar

        return force_N

    def current_force(self):
        return self._compute_force()

    def compute_train_dfdy(self, ud, train_state=None, brake_state=None):
        """
        Computes variation of the state w.r.t. to time.
        """

        if (ud < -1):
            ud = -1
        elif (ud > 1):
            ud = 1
        dfdy = np.array([0.0, 0.0, 0.0])
        if (train_state is None):
            train_state = self.state
        if (brake_state is None):
            brake_state = self.brake.state

        current_force = self._compute_force(train_state.v, train_state.p)
        dfdy[0] = train_state.v
        dfdy[1] = current_force/self.parameter.M - \
            g * np.sin(self.train_path.get_theta(train_state.xi))
        speed_modulo = fabs(train_state.v)
        propulsion = self.parameter.a + self.parameter.b * speed_modulo + \
            self.parameter.c * (speed_modulo ** 2)
        other_forces = brake_state.b + propulsion

        if (train_state.v > 0):
            # Remove brake force if v is greater than 0
            dfdy[1] -= 1/self.parameter.M * other_forces
        if (train_state.v < 0):
            # Remove brake force if v is greater than 0
            dfdy[1] += 1/self.parameter.M * other_forces

        if (((train_state.p <= self.parameter.plowerbar) and (ud < 0)) or
           ((train_state.p >= self.parameter.pbar) and (ud > 0))):
            dfdy[2] = 0.0
        else:
            dfdy[2] = self.parameter.alpha * ud

        train_state_variation = train_state.deep_copy()
        train_state_variation.xi = dfdy[0]
        train_state_variation.v = dfdy[1]
        train_state_variation.p = dfdy[2]

        return train_state_variation

    def _compute_train_and_brake_dfdy(self, ud, ub, train_state=None,
                                      brake_state=None):
        train_state_variation = self.compute_train_dfdy(ud, train_state,
                                                        brake_state)
        brake_state_variation = self.brake.compute_dfdy(ub, brake_state)
        system_state_variation = SystemState(train_state_variation,
                                             brake_state_variation)
        return system_state_variation

    def compute_train_and_brake_dfdy(self, ud, ub, system_state):
        """
            Compute variations of train and brake states.

            For continuous states the computed values represents the variation
            over time (i.e., dx/dt). For discrete states, the variations
            represent the new values they assume.
        """

        return self._compute_train_and_brake_dfdy(ud, ub,
                                                  system_state.train_state,
                                                  system_state.brake_state)

    def compute_state_update(self, ud, ub, dt):
        """
        Computes the new state value. Code based on RK 4.
        """

        # First iteration
        system_state_1 = SystemState(self.state, self.brake.state)
        state_variation_1 = \
            self.compute_train_and_brake_dfdy(ud, ub, system_state_1)

        # Second iteration
        system_state_2 = self.update_given_system_state(
            system_state_1, state_variation_1, coefficient=dt/2.0)

        state_variation_2 = \
            self.compute_train_and_brake_dfdy(ud, ub, system_state_2)

        # Third iteration
        system_state_3 = \
            self.update_given_system_state(system_state_2, state_variation_2,
                                           coefficient=dt/2.0)

        state_variation_3 = \
            self.compute_train_and_brake_dfdy(ud, ub, system_state_3)

        # Fourth iteration
        system_state_4 = \
            self.update_given_system_state(system_state_3, state_variation_3,
                                           coefficient=dt)
        state_variation_4 = \
            self.compute_train_and_brake_dfdy(ud, ub, system_state_4)

        # Compose partial sums
        part1 = \
            self.update_given_system_state(system_state_1, state_variation_1,
                                           coefficient=dt/6.0)

        part2 = \
            self.update_given_system_state(part1, state_variation_2,
                                           coefficient=dt/3.0)

        part3 = \
            self.update_given_system_state(part2, state_variation_3,
                                           coefficient=dt/3.0)

        final_system_state = \
            self.update_given_system_state(part3, state_variation_4,
                                           coefficient=dt/6.0)

        return final_system_state

    def compute_and_set_new_state(self, ud, ub, dt):
        new_system_state = self.compute_state_update(ud, ub, dt)
        self.state = new_system_state.train_state
        self.brake.state = new_system_state.brake_state
        return new_system_state

    def update_given_state(self, train_state, deltaxi, deltav, deltap):
        """ Returns the updated value of the train state."""
        updated_state = train_state.deep_copy()
        updated_state.xi += deltaxi
        updated_state.v += deltav
        updated_state.p += deltap
        updated_state.p = min(updated_state.p, self.parameter.pbar)  # cap d
        updated_state.p = max(updated_state.p, self.parameter.plowerbar)
        return updated_state

    def update_given_system_state(self, system_state, delta_state,
                                  coefficient):
        """ Returns the updated value of the train and brake state."""

        deltaxi = delta_state.train_state.xi * coefficient
        deltav = delta_state.train_state.v * coefficient
        deltap = delta_state.train_state.p * coefficient
        deltab = delta_state.brake_state.b * coefficient
        delta_b_max = delta_state.brake_state.bMax * coefficient
        new_discrete_state = delta_state.brake_state.discrete

        updated_train_state = self.update_given_state(system_state.train_state,
                                                      deltaxi, deltav,
                                                      deltap)

        updated_brake_state = \
            self.brake.update_given_state(system_state.brake_state,
                                          deltab, delta_b_max,
                                          new_discrete_state)

        return SystemState(updated_train_state, updated_brake_state)
