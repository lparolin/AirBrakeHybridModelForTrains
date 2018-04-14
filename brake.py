"""
Defines the brake and its parameters.

"""

import numpy as np


class BrakeParameter(object):
    """
    Stores the parameters of a hydraulic brake
    """

    def __init__(self, beta=1230, gamma=617,
                 delta=3e3, bbar=3.7e+5):
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.bbar = bbar


class DiscreteValue(object):
    Idle, Brake, Release = range(3)


class BrakeState(object):
    """
    Stores the state of a brake.
    """

    def __init__(self, b0=0, bMax0=1.8519e+10,
                 discrete_state0=DiscreteValue.Release):
        self.b = b0
        self.bMax = bMax0
        self.discrete = discrete_state0

    def deep_copy(self):
        new_state = BrakeState()
        new_state.b = self.b
        new_state.bMax = self.bMax
        new_state.discrete = self.discrete
        return new_state

    def __str__(self):
        string_to_return = """Brake force: %.3g (N)
        Current max braking force: %.3g (N)
        Discrete state: %s""" % (self.b, self.bMax, self.discrete)
        return string_to_return


class Brake:
    """
    Models a hydraulic train brake.
    """

    def __init__(self, initial_state, parameter):
        self.state = initial_state
        self.parameter = parameter

    def compute_dfdy(self, ub, state=None):
        """Return the value of db/dt and dbMax/dt based on provided state."""

        if (ub > 1):
            ub = 1
        elif (ub < -1):
            ub = -1
        dfdy = np.array([0.0, 0.0])
        if (state is None):
            state = self.state

        if (state.discrete == DiscreteValue.Idle):
            if (ub > 0):
                # Call this routine again, with a different value
                new_state = state.deep_copy()
                new_state.discrete = DiscreteValue.Brake
                return self.compute_dfdy(ub, new_state)
            else:
                dfdy[0] = 0.0
                if (state.bMax < self.parameter.bbar):
                    dfdy[1] = self.parameter.beta
                else:
                    dfdy[1] = 0

        elif (state.discrete == DiscreteValue.Brake):
            if (ub < 0):
                new_state = state.deep_copy()
                new_state.discrete = DiscreteValue.Release
                return self.compute_dfdy(ub, new_state)
            else:
                if (state.b < state.bMax):
                    dfdy[0] = self.parameter.gamma * ub
                else:
                    dfdy[0] = 0
        # Release state
        else:
            if (state.b <= 0):
                new_state = state.deep_copy()
                new_state.discrete = DiscreteValue.Idle
                new_state.b = 0
                new_state.bMax = 0  # Set max breaking to 0
                return self.compute_dfdy(ub, new_state)
            else:
                dfdy[0] = -1.0 * self.parameter.delta
                dfdy[1] = -1.0 * self.parameter.delta
        state_variation = BrakeState()
        state_variation.b = dfdy[0]
        state_variation.bMax = dfdy[1]
        state_variation.discrete = state.discrete
        return state_variation

    def update_given_state(self, brake_state, deltab, delta_b_max,
                           new_discrete):
        """
        Returns the updated brake state given input and delta.
        """

        new_state = brake_state.deep_copy()
        new_state.discrete = new_discrete

        if (new_state.discrete == DiscreteValue.Idle):
            new_state.bMax += delta_b_max
            new_state.b = 0
            if (new_state.bMax >= self.parameter.bbar):
                new_state.bMax = self.parameter.bbar
            return new_state

        elif (new_state.discrete == DiscreteValue.Brake):
            new_state.b += deltab
            new_state.b = min(new_state.b, new_state.bMax)
            if (new_state.b <= 0):
                new_state.b = 0
            return new_state
        else:
            # We are in the Release state
            new_state.b += deltab
            new_state.bMax = 0
            if (new_state.b <= 0):
                new_state.b = 0
                new_state.discrete = DiscreteValue.Idle
            return new_state
