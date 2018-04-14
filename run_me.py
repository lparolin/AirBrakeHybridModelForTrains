"""
Main script for running the simulation.

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from trainpath import TrainPath
from train import Train, TrainParameter, TrainState
from brake import Brake, BrakeParameter, BrakeState, DiscreteValue
from scipy.constants import g


def vbar(train_position):
    """
    Define the desired speed of the train in (m/s) as a function of the
    position.
    """
    first_part_idx = np.logical_and((train_position >= 0.0),
                                    (train_position < (L/10.0)))
    nominal_speed_idx = np.logical_and((train_position >= (L/10.0)),
                                       (train_position < (L/10.0 * 9.0)))
    decrease_speed_idx = np.logical_and((train_position >= 9.0/10.0*L),
                                        (train_position < L))

    speed_profile = np.zeros(train_position.size)
    vmin = vNominal / 3  # min speed of the train
    speed_profile[first_part_idx] = (vNominal - vmin) * 10.0 * \
        train_position[first_part_idx]/L + vmin
    speed_profile[nominal_speed_idx] = vNominal
    speed_profile[decrease_speed_idx] = vNominal * 10.0 * \
        (1 - train_position[decrease_speed_idx] / L)
    return speed_profile


def vhat(train_position):
    """
    Define the maximum speed of the train in (m/s) as a function of the
    position.
    """
    # Take desired profile speed and add some buffer only if lower than 100
    speed_profile = np.minimum(vbar(train_position) + 30 / 3.6,
                               np.ones(train_position.size) * 100.0/3.6)
    return speed_profile


###############################
# Main script starts here

# Close all previous-generated graphs
plt.close("all")

# Create the path
myPath = TrainPath()

# Define the brake
brake_parameter = BrakeParameter()
brake_state = BrakeState(b0=0, bMax0=brake_parameter.bbar)
brake = Brake(brake_state, brake_parameter)

# Train
train_parameter = TrainParameter()
train_state = TrainState()
train = Train(train_parameter, train_state, brake, myPath)

# Simulation parameters
L = myPath.L
dt = 10                # [s] sample time
vNominal = 70.0 / 3.6  # nominal speed [m/s]
maxTime = 4 * 3600     # (of the simulation, [s])

use_minutes_for_time_in_plot = True  # Set to False to show time in hours

speed_threshold = 10 / 3.6  # (threshold for dummy controller, m/s)

# Simulation-related variables
time_step = np.arange(0, maxTime + dt, dt)
n_steps = time_step.size

train_speed = np.zeros(n_steps)      # Store reported train speed over time
train_position = np.zeros(n_steps)   # Store reported train position over time
engine_power = np.zeros(n_steps)     # Store engine-generated force
engine_force = np.zeros(n_steps)     # Store engine-generated force
brake_force = np.zeros(n_steps)      # Store brake-generated force
max_brake_force = np.zeros(n_steps)  # Store max available brake force
# Discrete state of the brake
discrete_brake_state = np.zeros(n_steps, dtype='int')
engine_command = np.zeros(n_steps)   # Input signal to the engine
brake_command = np.zeros(n_steps)    # Input signal to the brake

# Store initial state
train_position[0] = train.state.xi
train_speed[0] = train.state.v
engine_power[0] = train.state.p
engine_force[0] = train.current_force()
brake_force[0] = brake.state.b
max_brake_force[0] = brake.state.bMax
discrete_brake_state[0] = brake.state.discrete

brake_input = 0
drive_input = 1
for i_step in range(1, n_steps):

    # Example of a control strategy
    if (train.state.v < (vbar(np.array([train.state.xi])) - speed_threshold)):
        drive_input = 1
        brake_input = -1
    elif (train.state.v > (vbar(np.array([train.state.xi])) +
          speed_threshold)):
        if (train.state.v > (vbar(np.array([train.state.xi])) +
           5 * speed_threshold)):
            # brake with engine and air brake
            drive_input = -1
            brake_input = 1
        else:
            # just engine should be sufficient
            drive_input = -1
            brake_input = 0
    else:
        drive_input = 0
        brake_input = -1

    new_system_state = train.compute_and_set_new_state(drive_input,
                                                       brake_input, dt)
    train_speed[i_step] = new_system_state.train_state.v
    train_position[i_step] = new_system_state.train_state.xi
    engine_power[i_step] = new_system_state.train_state.p
    engine_force[i_step] = train.current_force()
    brake_force[i_step] = new_system_state.brake_state.b
    discrete_brake_state[i_step] = new_system_state.brake_state.discrete
    max_brake_force[i_step] = new_system_state.brake_state.bMax
    engine_command[i_step] = drive_input
    brake_command[i_step] = brake_input

# In this section we compute some relevant variables for the simulation
desired_speed_over_time = vbar(train_position)
max_speed_over_time = vhat(train_position)
y_over_time = myPath.get_y_coordinate(train_position)
angle_over_time = myPath.get_theta(train_position)

speed_modulo = np.absolute(train_speed)
mu = train.parameter.a + train.parameter.b * speed_modulo + \
    (speed_modulo ** 2) * train.parameter.c

grativational_force = -train.parameter.M * g * \
    np.sin(train.train_path.get_theta(train_position))

# Plotting section
if (use_minutes_for_time_in_plot):
    time_x = time_step / 60.0
    xlabel_time = "Time (min)"
else:
    time_x = time_step / 3600.0  # hours
    xlabel_time = "Time (hr)"
M = train_parameter.M

# color definition
c_xi = 'black'
c_v = 'green'
c_vhat = 'red'
c_vbar = 'gray'
y_vbar = 'blue'

c_ef, c_b, c_maxb, c_mu, c_g = sns.husl_palette(5)
# color values and names from http://xkcd.com/color/rgb/
c_ec = '#029386'  # teal
c_bc = '#c20078'  # magenta

fig, ax = plt.subplots(4, sharex=True)
ax[0].plot(time_x, train_position * 1.0e-3, label='xi(t)', color=c_xi)
ax[1].plot(time_x, train_speed * 3.6, label='v(t)', color=c_v)
ax[1].plot(time_x, desired_speed_over_time * 3.6, label='vbar(t)',
           linewidth=2.0, color=c_vbar)
ax[1].plot(time_x, max_speed_over_time * 3.6, label='vhat(t)',
           linewidth=2.0, color=c_vhat)
ax[2].plot(time_x, y_over_time, label='Altitude')
ax[3].plot(time_x, engine_force/1e3, label='f(t)', color=c_ef)
ax[3].plot(time_x, brake_force/1e3, label='b(t)', color=c_b)
ax[3].plot(time_x, mu/1e3, label='mu(t)', color=c_mu)
ax[3].set_xlabel(xlabel_time)
ax[1].set_ylabel("Speed (km/h)")
ax[0].set_ylabel("Position (km)")
ax[2].set_ylabel("Altitude (m)")
ax[3].set_ylabel("Force (kN)")
ax[1].legend()
ax[3].legend()
fig.savefig('fig1.png', bbox_inches='tight')

fig, ax = plt.subplots(2, sharex=True)
ax[0].plot(time_x, engine_force * 1e-3, label='d', color=c_ef)
ax[0].plot(time_x, brake_force * 1e-3, label='b', color=c_b)
ax[0].plot(time_x, mu * 1e-3, label='mu', color=c_mu)
ax[0].plot(time_x, grativational_force * 1e-3, label='gravity', color=c_g)
ax[1].plot(time_x, engine_command, label='ud', color=c_ec)
ax[1].plot(time_x, brake_command, label='ub', color=c_bc)
ax[0].legend()
ax[1].legend()
ax[0].set_ylabel("Force. (kN)")
ax[1].set_ylabel("Control values")
ax[1].set_ylim([-1.2, 1.2])
ax[1].set_xlabel(xlabel_time)
fig.savefig('fig2.png', bbox_inches='tight')

fig, ax = plt.subplots(3, sharex=True)
ax[0].plot(time_x, brake_force * 1e-3, label='b', color=c_b)
ax[0].plot(time_x, max_brake_force * 1e-3, label='bMax', color=c_maxb)
ax[1].plot(time_x, discrete_brake_state, label='state(t)')
ax[2].plot(time_x, brake_command, label='ub', color=c_bc)

ax[0].legend()
ax[0].set_ylabel("Braking forces (kN)")
ax[2].set_ylabel("ub(t)")
ax[2].set_xlabel(xlabel_time)

# Some more work for ax[1]
ax[1].set_yticks([0, 1, 2])
ax[1].set_ylim([-0.1, 2.1])

fig.canvas.draw()
labels = [item.get_text() for item in ax[1].get_yticklabels()]
# Change labels from numeric to string
for i_label in range(size(labels)):
    f_value = float(labels[i_label])
    if f_value == float(DiscreteValue.Idle):
        labels[i_label] = 'Idle'
    elif f_value == float(DiscreteValue.Brake):
        labels[i_label] = 'Brake'
    elif f_value == float(DiscreteValue.Release):
        labels[i_label] = 'Release'
    else:
        labels[i_label] = ''

ax[1].set_yticklabels(labels)
fig.savefig('fig3.png', bbox_inches='tight')

plt.show()
