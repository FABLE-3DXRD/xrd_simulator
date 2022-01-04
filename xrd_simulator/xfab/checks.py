import numpy as np

"""The checks module implements input checks for various xfab functions. These checks
are meant to be called upon during runtime to catch errors made on the user side. The checks
module is in its default mode active, however, by toogling a module level variable all check
across xfab can be easily turned on or of. This allows users to in some specific scenarios
speed up their codes. Furthermore, if __debug__==False all checks are inactivated, i.e
using python -O to run codes will remove any checks. Example usage

import xfab

xfab.checks.is_activated()

out: True

xfab.off()
xfab.checks.is_activated()

out: False

xfab.on()
xfab.checks.is_activated()

out: True

"""

_verify = True


def on():
    """Turn on flag indicating if checks on input and output from xrd_simulator.xfab functions should be run.
    """
    _verify = True


def off():
    """Turn off flag indicating if checks on input and output from xrd_simulator.xfab functions should be run.
    """
    _verify = False


def is_activated():
    """Return True if checks are to be run.
    """
    return _verify and __debug__


def _check_rotation_matrix(U):
    """Verify that a 3 x 3 matrix is a rotation matrix.

    Args:
        U: 3x3 matrix represented as a numpy array of shape=(3,3).

    """
    if not np.allclose(np.dot(U.T, U), np.eye(3, 3)):
        raise ValueError(
            "orientation matrix U is not unitary, np.dot(U.T, U)!=np.eye(3,3)")

    if not np.allclose(np.linalg.det(U), 1.0):
        raise ValueError(
            "orientation matrix U has a non unity determinant np.linalg.det(U)!=1.0")


def _check_euler_angles(phi1, PHI, phi2):
    """Verify that all three Euler angles lies in the range [0,2*pi].

    Args:
        phi1, PHI, and phi2: Euler angles in radians.

    """
    if not (0 <= phi1 <= np.pi * 2):
        raise ValueError(
            "Euler angle phi1=" +
            str(phi1) +
            " is not in range [0,2*pi]")

    if not (0 <= PHI <= np.pi * 2):
        raise ValueError(
            "Euler angle PHI=" +
            str(PHI) +
            " is not in range [0,2*pi]")

    if not (0 <= phi2 <= np.pi * 2):
        raise ValueError(
            "Euler angle phi2=" +
            str(phi2) +
            " is not in range [0,2*pi]")
