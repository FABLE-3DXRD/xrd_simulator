import numpy as np


class RigidBodyMotion(object):
    """Rigid body transform euclidean points by an euler axis and translation representation.

    The Motion is parametric in the interval time=[0,1] and will perform a rigid body transformation
    of a point x by linearly uniformly rotating it from [0, rotation_angle] and translating [0, translation].
    I.e if called at a time time=t the motion will first rotate the point ``t*rotation_angle`` radians
    around ``rotation_axis`` and next translate the point by the vector ``t*translation``.

    Args:
        rotation_axis (:obj:`numpy array`): Rotation axis ``shape=(3,)``
        rotation_angle (:obj:`float`): Radians for final rotation, when time=1.
        translation (:obj:`numpy array`):  Translation vector ``shape=(3,)``

    Attributes:
        rotation_axis (:obj:`numpy array`): Rotation axis ``shape=(3,)``
        rotation_angle (:obj:`float`): Radians for final rotation, when time=1.
        translation (:obj:`numpy array`):  Translation vector ``shape=(3,)``

    """

    def __init__(self, rotation_axis, rotation_angle, translation):
        assert rotation_angle < np.pi and rotation_angle > 0, "The rotation angle must be in [0 pi]"
        self.rotator = _RodriguezRotator(rotation_axis)
        self.rotation_axis = rotation_axis
        self.rotation_angle = rotation_angle
        self.translation = translation

    def rotate(self, vectors, time):
        """Find the rotational transformation of a set of vectors at a prescribed time.

        NOTE: This function only applies the rigid body rotation

        Args:
            vectors (:obj:`numpy array`): A set of points in 3d euclidean space to be rotated (``shape=(3,N)``)
            time (:obj:`float`): Time to compute for.

        Returns:
            Transformed vectors (:obj:`numpy array`) of ``shape=(3,N)``.

        """
        assert time <= 1 and time >= 0, "The rigid body motion is only valid on the interval time=[0,1]"
        return self.rotator(vectors, self.rotation_angle * time)

    def translate(self, vectors, time):
        """Find the translational transformation of a set of points at a prescribed time.

        NOTE: This function only applies the rigid body translation

        Args:
            vectors (:obj:`numpy array`): A set of points in 3d euclidean space to be rotated (``shape=(3,N)``)
            time (:obj:`float`): Time to compute for.

        Returns:
            Transformed vectors (:obj:`numpy array`) of ``shape=(3,N)``.

        """
        assert time <= 1 and time >= 0, "The rigid body motion is only valid on the interval time=[0,1]"
        if len(vectors.shape) > 1:
            translation = self.translation.reshape(3, 1)
        else:
            translation = self.translation
        return vectors + translation * time

    def __call__(self, vectors, time):
        """Find the transformation of a set of points at a prescribed time.

        Args:
            vectors (:obj:`numpy array`): A set of points in 3d euclidean space to be rotated (``shape=(3,N)``)
            time (:obj:`float`): Time to compute for.

        Returns:
            Transformed vectors (:obj:`numpy array`) of ``shape=(3,N)``.

        """
        assert time <= 1 and time >= 0, "The rigid body motion is only valid on the interval time=[0,1]"
        if len(vectors.shape) > 1:
            translation = self.translation.reshape(3, 1)
        else:
            translation = self.translation
        return self.rotator(vectors, self.rotation_angle *
                            time) + translation * time


class _RodriguezRotator(object):
    """Object for rotating vectors in the plane described by yhe unit normal rotation_axis.

    Args:
        rotation_axis (:obj:`numpy array`): A unit vector in 3d euclidean space (``shape=(3,)``)

    Attributes:
        rotation_axis (:obj:`numpy array`): A unit vector in 3d euclidean space (``shape=(3,)``)
        K (:obj:`numpy array`): (``shape=(3,3)``)
        K2 (:obj:`numpy array`): (``shape=(3,3)``)
        I (:obj:`numpy array`): (``shape=(3,3)``)

    """

    def __init__(self, rotation_axis):
        assert np.allclose(np.linalg.norm(rotation_axis),
                           1), "The rotation axis must be length unity."
        self.rotation_axis = rotation_axis
        rx, ry, rz = self.rotation_axis
        self.K = np.array([[0, -rz, ry],
                           [rz, 0, -rx],
                           [-ry, rx, 0]])
        self.K2 = self.K.dot(self.K)

    def get_rotation_matrix(self, rotation_angle):
        return np.eye(3, 3) + np.sin(rotation_angle) * self.K + \
            (1 - np.cos(rotation_angle)) * self.K2

    def __call__(self, vectors, rotation_angle):
        """Rotate a vector in the plane described by v1 and v2 towards v2 a fraction s=[0,1].

        Args:
            vectors (:obj:`numpy array`): A set of vectors in 3d euclidean space to be rotated (``shape=(3,N)``)
            rotation_angle (:obj:`float`): Radians to rotate vectors around the rotation_axis (positive rotation).

        Returns:
            Rotated vectors (:obj:`numpy array`) of ``shape=(3,N)``.

        """
        R = self.get_rotation_matrix(rotation_angle)
        return R.dot(vectors)
