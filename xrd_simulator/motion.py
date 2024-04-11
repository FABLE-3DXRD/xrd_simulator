"""The motion module is used to represent a rigid body motion. During diffraction from a
:class:`xrd_simulator.polycrystal.Polycrystal` the :class:`xrd_simulator.motion.RigidBodyMotion`
object describes how the sample is translating and rotating. The motion can be used to update the
polycrystal position via the :func:`xrd_simulator.polycrystal.Polycrystal.transform` function.

Here is a minimal example of how to instantiate a rigid body motion object, apply the motion to a pointcloud
and save the motion to disc:

    Examples:
        .. literalinclude:: examples/example_init_motion.py

Below follows a detailed description of the RigidBodyMotion class attributes and functions.

"""
import numpy as np
import dill

class RigidBodyMotion():
    """Rigid body transformation of euclidean points by an euler axis rotation and a translation.

    A rigid body motion is defined in the laboratory coordinates system.

    The Motion is parametric in the interval time=[0,1] and will perform a rigid body transformation
    of a point x by linearly uniformly rotating it from [0, rotation_angle] and translating [0, translation].
    I.e if called at a time time=t the motion will first rotate the point ``t*rotation_angle`` radians
    around ``rotation_axis`` and next translate the point by the vector ``t*translation``.

    Args:
        rotation_axis (:obj:`numpy array`): Rotation axis ``shape=(3,)``
        rotation_angle (:obj:`float`): Radians for final rotation, when time=1.
        translation (:obj:`numpy array`):  Translation vector ``shape=(3,)``
        origin (:obj:`numpy array`): Point in space about which the rigid body motion is
            defined Defaults to the origin (0,0,0). All translations are executed in relation
            to the origin and all rotation are rotations about the point of origin. ``shape=(3,)``

    Attributes:
        rotation_axis (:obj:`numpy array`): Rotation axis ``shape=(3,)``
        rotation_angle (:obj:`float`): Radians for final rotation, when time=1.
        translation (:obj:`numpy array`):  Translation vector ``shape=(3,)``
        origin (:obj:`numpy array`): Point in space about which the rigid body motion is
            defined Defaults to the origin (0,0,0). All translations are executed in relation
            to the origin and all rotation are rotations about the point of origin. ``shape=(3,)``

    """

    def __init__(self, rotation_axis, rotation_angle, translation, origin=np.zeros((3,))):
        assert rotation_angle < np.pi and rotation_angle > 0, "The rotation angle must be in [0 pi]"
        self.rotator = _RodriguezRotator(rotation_axis)
        self.rotation_axis = rotation_axis
        self.rotation_angle = rotation_angle
        self.translation = translation
        self.origin = origin

    def __call__(self, vectors, time):
        """Find the transformation of a set of points at a prescribed time.

        Calling this method will execute the rigid body motion with respect to the
            currently set origin.

        Args:
            vectors (:obj:`numpy array`): A set of points to be transformed (``shape=(3,N)``)
            time (:obj:`float`): Time to compute for.

        Returns:
            Transformed vectors (:obj:`numpy array`) of ``shape=(3,N)``.

        """
        #assert time <= 1 and time >= 0, "The rigid body motion is only valid on the interval time=[0,1]"
        
        if len(vectors.shape) == 1:
            translation = self.translation
            origin = self.origin
            centered_vectors = vectors - origin
            centered_rotated_vectors  =  self.rotator(centered_vectors, self.rotation_angle * time)
            rotated_vectors = centered_rotated_vectors + origin
            return rotated_vectors + translation * time
        
        elif len(vectors.shape) == 2:
            translation = self.translation.reshape(1,3)
            origin = self.origin.reshape(1,3)      
            centered_vectors = vectors - origin
            centered_rotated_vectors  =  self.rotator(centered_vectors, self.rotation_angle * time)
            rotated_vectors = centered_rotated_vectors + origin
            if np.isscalar(time):
                return rotated_vectors + translation * time
            return rotated_vectors + translation * np.array(time)[:,np.newaxis]
        
        elif len(vectors.shape) == 3:
            translation = self.translation.reshape(1,3)
            origin = self.origin.reshape(1,3)
            centered_vectors = vectors - origin
            centered_rotated_vectors  =  self.rotator(centered_vectors.reshape(-1,3), self.rotation_angle * np.tile(time,(4,1)).T.reshape(-1)).reshape(-1,4,3)
            rotated_vectors = centered_rotated_vectors + origin       
            return rotated_vectors + translation * np.array(time)[:,np.newaxis,np.newaxis]
    
    def rotate(self, vectors, time):
        """Find the rotational transformation of a set of vectors at a prescribed time.

        NOTE: This function only applies the rigid body rotation and will not respect the
            origin of the motion! This function is intended for rotation of diffraction
            and wavevectors. Use the __call__ method to perform a physical rigidbody motion
            respecting the origin.

        Args:
            vectors (:obj:`numpy array`): A set of points in 3d euclidean space to be rotated (``shape=(3,N)``)
            time (:obj:`float`): Time to compute for.

        Returns:
            Transformed vectors (:obj:`numpy array`) of ``shape=(3,N)``.

        """
        #assert time <= 1 and time >= 0, "The rigid body motion is only valid on the interval time=[0,1]"
        rotated_vectors  = self.rotator(vectors, self.rotation_angle * time)
        return rotated_vectors

    def translate(self, vectors, time):
        """Find the translational transformation of a set of points at a prescribed time.

        NOTE: This function only applies the rigid body translation.

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

    def inverse(self):
        """Create an instance of the inverse motion, defined by negative translation- and rotation-axis vectors.

        Returns:
            (:obj:`xrd_simulator.RigidBodyMotion`) The inverse motion with a reversed rotation and translation.

        """
        return RigidBodyMotion(-self.rotation_axis.copy(), self.rotation_angle, -self.translation.copy(), self.origin.copy())

    def save(self, path):
        """Save the motion object to disc (via pickling).

        Args:
            path (:obj:`str`): File path at which to save, ending with the desired filename.

        """
        if not path.endswith(".motion"):
            path = path + ".motion"
        with open(path, "wb") as f:
            dill.dump(self, f, dill.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        """Load the motion object from disc (via pickling).

        Args:
            path (:obj:`str`): File path at which to load, ending with the desired filename.

        .. warning::
            This function will unpickle data from the provied path. The pickle module
            is not intended to be secure against erroneous or maliciously constructed data.
            Never unpickle data received from an untrusted or unauthenticated source.

        """
        if not path.endswith(".motion"):
            raise ValueError("The loaded motion file must end with .motion")
        with open(path, 'rb') as f:
            return dill.load(f)

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
        return (np.eye(3, 3)[:,:,np.newaxis] + np.sin(rotation_angle) * self.K[:,:,np.newaxis] + (1 - np.cos(rotation_angle)) * self.K2[:,:,np.newaxis]).transpose(2,0,1)

    def __call__(self, vectors, rotation_angle):
        """Rotate a vector in the plane described by v1 and v2 towards v2 a fraction s=[0,1].

        Args:
            vectors (:obj:`numpy array`): A set of vectors in 3d euclidean space to be rotated (``shape=(3,N)``)
            rotation_angle (:obj:`float`): Radians to rotate vectors around the rotation_axis (positive rotation).

        Returns:
            Rotated vectors (:obj:`numpy array`) of ``shape=(3,N)``.

        """

        R = self.get_rotation_matrix(rotation_angle)
        
        return np.matmul(R,vectors[:,:,np.newaxis])[:,:,0] # Syntax valid for the rotation fo the G vectors from the grains
