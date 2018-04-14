"""
This class defines and stores information about the path of the train.

"""

import numpy as np
from math import pi as PI
import matplotlib.pyplot as plt
import prettyplotlib as ppl


class TrainPath(object):
    """
    Stores and defines information about the path of the train.

    X,Y coordinate at every point of the path are stores as well as the
    angle of the path at any given point xi.
    """

    _DEF_SEGMENT_LENGTH_AND_ANGLE = np.array(
        [[2,    4,  3,   6, 4, 1, 3,    1, 4,  5,  3,  4.0, 2.0],
         [0, -1.0,  0, 0.4, 0, 1, 0, -0.5, 0, 0.5, 0, -1.0, 0.0]])

    _x_interp = 0
    _y_interp = 0
    _theta_interp = 0

    def __init__(self, path_total_length=200.0e3,
                 segment_length_and_angle=None):
        if (segment_length_and_angle is None):
            segment_length_and_angle = self._DEF_SEGMENT_LENGTH_AND_ANGLE

        segment_length_rel = segment_length_and_angle[0]
        segment_length = segment_length_rel / np.sum(segment_length_rel) * \
            path_total_length
        angle_array_rad = segment_length_and_angle[1] * PI / 180.0

        segment_height = segment_length * np.sin(angle_array_rad)
        segment_x_length = segment_length * np.cos(angle_array_rad)
        self._y_coordinate = np.append([0.0], np.cumsum(segment_height))
        self._x_coordinate = np.append([0.0], np.cumsum(segment_x_length))
        self._l_coordinate = np.append([0.0], np.cumsum(segment_length))
        self._theta_rad = angle_array_rad

        self._get_x = lambda x: np.interp(x, self._l_coordinate,
                                          self._x_coordinate)
        self._get_y = lambda x: np.interp(x, self._l_coordinate,
                                          self._y_coordinate)
        self.L = path_total_length

    def get_theta(self, length_array):
        """
        Returns the angle of the path in rad.

        Parameters:
           length_array: np.array, positions (in meter) at which the angle
                         will be computed
        """

        if (type(length_array).__module__ != np.__name__):
            # Cast into a length_array
            length_array = np.array([length_array])

        if (length_array.size == 1):
            # different code for scalar
            # Find the lower part
            idx_lower = self._l_coordinate[1:] <= length_array
            if all(idx_lower == True):
                return self._theta_rad[-1]  # return last element
            elif all(idx_lower == False):
                return self._theta_rad[0]   # return first element
            else:
                # Take the index associated with the first false element
                bigger_theta = self._theta_rad[np.logical_not(idx_lower)]
                return bigger_theta[0]  # only first element

        theta = np.zeros(length_array.size)
        for i_segment in range(1, len(self._l_coordinate)):
            idx_to_take = np.logical_and(
                (length_array >= self._l_coordinate[i_segment-1]),
                (length_array < self._l_coordinate[i_segment]))
            if (idx_to_take.size == 0):
                continue
            else:
                theta[idx_to_take] = self._theta_rad[i_segment - 1]

        # Fix the vaulues at the end of the sequence
        idx_to_take = (length_array == self._l_coordinate[-1])
        if (length_array.size == 1):
            return self._theta_rad[-1]
        theta[idx_to_take] = self._theta_rad[-1]
        return theta

    def get_x_coordinate(self, length_array):
        """
        Returns the x coordinate of the path. Only used for plotting the path
        on a x,y plot
        """
        return self._get_x(length_array)

    def get_y_coordinate(self, length_array):
        """
        Retuns the y coordinate of the path.
        """

        return self._get_y(length_array)

    def get_l_coordinate(self):
        """
        Returns a vector of points which denotes change of angle in the path.
        """
        return self._l_coordinate

    def plot_xy_path(self, is_xy_same_scale=False):
        """
        Plots the profile of the path in a x,y coordinate system.
        """

        fig, ax = plt.subplots(1)
        if (is_xy_same_scale):
            ppl.plot(ax, self._x_coordinate, self._y_coordinate)
            ax.axis('equal')
            ax.set_xlabel("X coordinate (m)")
        else:
            ppl.plot(ax, self._x_coordinate * 1.0e-3, self._y_coordinate)
            ax.set_xlabel("X coordinate (km)")
        ax.set_ylabel("Y coordinate (m)")
        ax.grid(axis='y')
        if (is_xy_same_scale):
            ax.axis('equal')
        fig.show()
        return (fig, ax)
