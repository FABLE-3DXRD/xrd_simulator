"""The motion module is used to represent a rigid body motion.

During diffraction from a :class:`xrd_simulator.polycrystal.Polycrystal`, the
:class:`xrd_simulator.motion.RigidBodyMotion` object describes how the sample
is translating and rotating. The motion can be used to update the polycrystal
position via the :func:`xrd_simulator.polycrystal.Polycrystal.transform`
function.

Examples
--------
Here is a minimal example of how to instantiate a rigid body motion object,
apply the motion to a pointcloud and save the motion to disc:

.. literalinclude:: examples/example_init_motion.py

Below follows a detailed description of the RigidBodyMotion class attributes
and functions.
"""

import dill
import torch
from xrd_simulator.utils import ensure_torch

torch.set_default_dtype(torch.float64)


class RigidBodyMotion:
    """Rigid body transformation by Euler axis rotation and translation.

    A rigid body motion is defined in the laboratory coordinates system.

    The motion is parametric in the interval ``time=[0, 1]`` and will perform a
    rigid body transformation of a point ``x`` by linearly uniformly rotating
    it from ``[0, rotation_angle]`` and translating ``[0, translation]``. I.e.,
    if called at a time ``time=t``, the motion will first rotate the point
    ``t * rotation_angle`` radians around ``rotation_axis`` and next translate
    the point by the vector ``t * translation``.

    Parameters
    ----------
    rotation_axis : numpy.ndarray or torch.Tensor
        Rotation axis, shape ``(3,)``.
    rotation_angle : float
        Radians for final rotation when ``time=1``.
    translation : numpy.ndarray or torch.Tensor
        Translation vector, shape ``(3,)``.
    origin : numpy.ndarray or torch.Tensor, optional
        Point in space about which the rigid body motion is defined. Defaults
        to the origin ``(0, 0, 0)``. All translations are executed in relation
        to the origin and all rotations are rotations about the point of origin.
        Shape ``(3,)``.

    Attributes
    ----------
    rotation_axis : torch.Tensor
        Rotation axis, shape ``(3,)``.
    rotation_angle : torch.Tensor
        Radians for final rotation when ``time=1``.
    translation : torch.Tensor
        Translation vector, shape ``(3,)``.
    origin : torch.Tensor
        Point in space about which the rigid body motion is defined,
        shape ``(3,)``.
    """

    def __init__(
        self, rotation_axis, rotation_angle, translation, origin=torch.zeros((3,))
    ):
        assert (
            rotation_angle < torch.pi and rotation_angle > 0
        ), "The rotation angle must be in [0 pi]"
        self.rotator = _RodriguezRotator(rotation_axis)
        self.rotation_axis = ensure_torch(rotation_axis)
        self.rotation_angle = ensure_torch(rotation_angle)
        self.translation = ensure_torch(translation)
        self.origin = ensure_torch(origin)

    def __call__(self, vectors, time):
        """Find the transformation of a set of points at a prescribed time.

        Calling this method executes the rigid body motion with respect to the
        currently set origin.

        Parameters
        ----------
        vectors : numpy.ndarray or torch.Tensor
            A set of points to be transformed, shape ``(3, N)``, ``(N, 3)``, or
            ``(N, 4, 3)``.
        time : float, numpy.ndarray, or torch.Tensor
            Time to compute for. Can be scalar or shape ``(N,)`` for per-vector
            times.

        Returns
        -------
        torch.Tensor
            Transformed vectors with shape ``(3, N)``, ``(N, 3)``, or
            ``(N, 4, 3)``.
        """
        # assert time <= 1 and time >= 0, "The rigid body motion is only valid on the interval time=[0,1]"
        vectors = ensure_torch(vectors)
        time = ensure_torch(time)

        if len(vectors.shape) == 1:
            translation = self.translation
            origin = self.origin
            centered_vectors = vectors - origin
            centered_rotated_vectors = self.rotator(
                centered_vectors, self.rotation_angle * time
            )
            rotated_vectors = centered_rotated_vectors + origin
            return torch.squeeze(rotated_vectors + translation * time)

        elif len(vectors.shape) == 2:
            translation = self.translation.reshape(1, 3)
            origin = self.origin.reshape(1, 3)
            centered_vectors = vectors - origin
            centered_rotated_vectors = self.rotator(
                centered_vectors, self.rotation_angle * time
            )
            rotated_vectors = centered_rotated_vectors + origin
            if time.ndim == 0 or (time.ndim == 1 and len(time) == 1):
                return rotated_vectors + translation * time

            return rotated_vectors + translation * time.unsqueeze(-1)

        elif len(vectors.shape) == 3:
            # Handle (N, M, 3) input shapes - typically (N_peaks, 4_vertices, 3_coords)
            N, M, _ = vectors.shape
            translation = self.translation.reshape(1, 3)
            origin = self.origin.reshape(1, 3)
            centered_vectors = vectors - origin
            centered_rotated_vectors = self.rotator(
                centered_vectors.reshape(-1, 3),
                self.rotation_angle * torch.tile(time, (M, 1)).T.reshape(-1),
            ).reshape(N, M, 3)
            rotated_vectors = centered_rotated_vectors + origin
            # Don't squeeze - preserve shape for proper indexing downstream
            return (
                rotated_vectors
                + translation * ensure_torch(time)[:, torch.newaxis, torch.newaxis]
            )

    def rotate(self, vectors, time):
        """Find the rotational transformation of a set of vectors.

        This function only applies the rigid body rotation and will not respect
        the origin of the motion. This function is intended for rotation of
        diffraction and wavevectors. Use the ``__call__`` method to perform a
        physical rigid body motion respecting the origin.

        Parameters
        ----------
        vectors : numpy.ndarray or torch.Tensor
            A set of points in 3D Euclidean space to be rotated, shape
            ``(3, N)`` or ``(N, 3)``.
        time : float, numpy.ndarray, or torch.Tensor
            Time to compute for. Can be scalar or shape ``(N,)`` for per-vector
            times.

        Returns
        -------
        torch.Tensor
            Transformed vectors of shape ``(3, N)`` or ``(N, 3)``.
        """
        # assert time <= 1 and time >= 0, "The rigid body motion is only valid on the interval time=[0,1]"
        time = ensure_torch(time)
        rotated_vectors = self.rotator(vectors, self.rotation_angle * time)
        return rotated_vectors

    def translate(self, vectors, time):
        """Find the translational transformation of a set of points.

        This function only applies the rigid body translation.

        Parameters
        ----------
        vectors : numpy.ndarray or torch.Tensor
            A set of points in 3D Euclidean space to be translated, shape
            ``(3, N)`` or ``(N, 3)``.
        time : float
            Time to compute for.

        Returns
        -------
        torch.Tensor
            Transformed vectors of shape ``(3, N)`` or ``(N, 3)``.
        """
        assert (
            time <= 1 and time >= 0
        ), "The rigid body motion is only valid on the interval time=[0,1]"
        
        vectors = ensure_torch(vectors)
        
        if len(vectors.shape) > 1:
            translation = self.translation.reshape(3, 1)
        else:
            translation = self.translation

        return vectors + translation * time

    def inverse(self):
        """Create an instance of the inverse motion.

        The inverse is defined by negative translation and rotation axis
        vectors.

        Returns
        -------
        RigidBodyMotion
            The inverse motion with a reversed rotation and translation.
        """
        return RigidBodyMotion(
            -self.rotation_axis.clone(),
            self.rotation_angle,
            -self.translation.clone(),
            self.origin.clone(),
        )

    def save(self, path):
        """Save the motion object to disc via pickling.

        Parameters
        ----------
        path : str
            File path at which to save, ending with the desired filename.
            The ``.motion`` extension is added if not present.
        """
        if not path.endswith(".motion"):
            path = path + ".motion"
        with open(path, "wb") as f:
            dill.dump(self, f, dill.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        """Load the motion object from disc via pickling.

        Parameters
        ----------
        path : str
            File path at which to load, ending with the desired filename.

        Returns
        -------
        RigidBodyMotion
            Loaded motion object.

        Raises
        ------
        ValueError
            If the file does not end with ``.motion``.

        Warnings
        --------
        This function will unpickle data from the provided path. The pickle
        module is not intended to be secure against erroneous or maliciously
        constructed data. Never unpickle data received from an untrusted or
        unauthenticated source.
        """
        if not path.endswith(".motion"):
            raise ValueError("The loaded motion file must end with .motion")
        with open(path, "rb") as f:
            return dill.load(f)


class _RodriguezRotator(object):
    """Object for rotating vectors around a unit normal rotation axis.

    Parameters
    ----------
    rotation_axis : numpy.ndarray or torch.Tensor
        A unit vector in 3D Euclidean space, shape ``(3,)``.

    Attributes
    ----------
    rotation_axis : torch.Tensor
        A unit vector in 3D Euclidean space, shape ``(3,)``.
    K : torch.Tensor
        Skew-symmetric cross-product matrix, shape ``(3, 3)``.
    K2 : torch.Tensor
        Square of the skew-symmetric matrix, shape ``(3, 3)``.
    """

    def __init__(self, rotation_axis):
        rotation_axis = ensure_torch(rotation_axis)
        assert torch.allclose(
            torch.linalg.norm(rotation_axis), ensure_torch(1.0)
        ), "The rotation axis must be length unity."
        self.rotation_axis = rotation_axis
        rx, ry, rz = self.rotation_axis
        self.K = ensure_torch([[0, -rz, ry], [rz, 0, -rx], [-ry, rx, 0]])
        self.K2 = torch.matmul(self.K, self.K)

    # def get_rotation_matrix(self, rotation_angle):
    #     """Get the rotation matrix for a given rotation angle."""
    #     identity_matrix = torch.eye(3, dtype=self.K.dtype).unsqueeze(2)
    #     sin_term = torch.sin(rotation_angle) * self.K.unsqueeze(2)
    #     cos_term = (1 - torch.cos(rotation_angle)) * self.K2.unsqueeze(2)

    #     rotation_matrix = identity_matrix + sin_term + cos_term
    #     rotation_matrix = rotation_matrix.permute(2, 0, 1)

    #     return rotation_matrix
    
    def get_rotation_matrix(self, rotation_angle):
        """Compute 3x3 rotation matrices for one or many rotation angles.

        Parameters
        ----------
        rotation_angle : torch.Tensor
            Scalar tensor or shape ``(N,)``.

        Returns
        -------
        torch.Tensor
            If scalar, returns shape ``(3, 3)``. If N angles, returns shape
            ``(N, 3, 3)``.
        """
        # Ensure rotation_angle is (..., 1, 1) for broadcasting
        # rotation_angle: scalar → shape (1,1,1)
        #                 vector (N,) → shape (N,1,1)
        rot = rotation_angle[..., None, None]

        # K and K2 are 3×3 tensors — add batch dims for broadcasting
        K  = self.K[None, :, :]      # (1,3,3)
        K2 = self.K2[None, :, :]     # (1,3,3)

        # Identity matrix, broadcastable
        I = torch.eye(3, dtype=self.K.dtype, device=self.K.device)[None, :, :]  # (1,3,3)

        # Rodrigues’ formula
        sin_term = torch.sin(rot) * K
        cos_term = (1 - torch.cos(rot)) * K2

        R = I + sin_term + cos_term  # shape: (N,3,3) or (1,3,3)

        # Return correct shape:
        # if input was scalar → return (3,3)
        if rotation_angle.ndim == 0:
            return R[0]
        else:
            return R


    def __call__(self, vectors, rotation_angle):
        """Rotate vectors around the rotation axis.

        Parameters
        ----------
        vectors : numpy.ndarray or torch.Tensor
            A set of vectors in 3D Euclidean space to be rotated, shape
            ``(N, 3)`` or ``(3,)`` for single vector.
        rotation_angle : float, numpy.ndarray, or torch.Tensor
            Radians to rotate vectors around the rotation axis (positive
            rotation). Can be scalar or shape ``(N,)`` for per-vector angles.

        Returns
        -------
        torch.Tensor
            Rotated vectors of shape ``(N, 3)`` or ``(3,)`` for single vector.
        """

        R = self.get_rotation_matrix(rotation_angle)
        vectors = ensure_torch(vectors)

        if len(vectors.shape) == 1:
            vectors = vectors[None, :]
        
        # Handle both scalar and vector rotation_angle cases
        if rotation_angle.ndim == 0:
            # Scalar case: R is (3,3), vectors is (N,3) -> output (N,3)
            return torch.squeeze(torch.matmul(R, vectors[:, :, None]))
        else:
            # Vector case: R is (N,3,3), vectors is (N,3) -> output (N,3)
            # Use bmm: (N,3,3) × (N,3,1) -> (N,3,1) -> squeeze to (N,3)
            return torch.bmm(R, vectors.unsqueeze(-1)).squeeze(-1)
