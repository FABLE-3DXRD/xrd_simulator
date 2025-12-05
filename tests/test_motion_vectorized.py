"""Test vectorized RigidBodyMotion with per-vector times for both rotation and translation."""

import unittest
import numpy as np
import torch
from xrd_simulator.motion import RigidBodyMotion


class TestVectorizedMotion(unittest.TestCase):
    """Test RigidBodyMotion with scalar and vector times."""

    def setUp(self):
        """Set up a simple motion for testing."""
        self.rotation_axis = np.array([0., 0., 1.])  # Rotate around z-axis
        self.rotation_angle = np.pi / 2.  # 90 degrees
        self.translation = np.array([1., 2., 3.])
        self.origin = np.array([0., 0., 0.])
        self.motion = RigidBodyMotion(
            self.rotation_axis,
            self.rotation_angle,
            self.translation,
            self.origin
        )

    def test_single_vector_scalar_time(self):
        """Test with single vector (3,) and scalar time."""
        vector = np.array([1., 0., 0.])
        time = 0.5
        
        result = self.motion(vector, time)
        
        # Should be shape (3,)
        self.assertEqual(result.shape, (3,))
        
        # At time=0.5, rotate 45 degrees around z-axis, then translate by 0.5*translation
        # [1,0,0] rotated 45° around z -> [cos(45°), sin(45°), 0] ≈ [0.707, 0.707, 0]
        # Then add [0.5, 1.0, 1.5]
        expected = np.array([
            np.cos(np.pi/4) + 0.5,  # ≈ 1.207
            np.sin(np.pi/4) + 1.0,   # ≈ 1.707
            0.0 + 1.5                # = 1.5
        ])
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5, atol=1e-10)

    def test_multiple_vectors_scalar_time(self):
        """Test with multiple vectors (N,3) and scalar time - all rotated/translated the same."""
        vectors = np.array([
            [1., 0., 0.],
            [0., 1., 0.],
            [1., 1., 0.]
        ])
        time = 1.0  # Full rotation and translation
        
        result = self.motion(vectors, time)
        
        # Should be shape (3, 3)
        self.assertEqual(result.shape, (3, 3))
        
        # At time=1.0, rotate 90° around z-axis, then translate by [1,2,3]
        # [1,0,0] -> [0,1,0] + [1,2,3] = [1,3,3]
        # [0,1,0] -> [-1,0,0] + [1,2,3] = [0,2,3]
        # [1,1,0] -> [-1,1,0] + [1,2,3] = [0,3,3]
        expected = np.array([
            [1., 3., 3.],
            [0., 2., 3.],
            [0., 3., 3.]
        ])
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5, atol=1e-10)

    def test_multiple_vectors_vector_times(self):
        """Test with multiple vectors (N,3) and per-vector times (N,) - each rotated/translated differently."""
        vectors = np.array([
            [1., 0., 0.],
            [1., 0., 0.],
            [1., 0., 0.]
        ])
        times = np.array([0.0, 0.5, 1.0])  # Different time for each vector
        
        result = self.motion(vectors, times)
        
        # Should be shape (3, 3)
        self.assertEqual(result.shape, (3, 3))
        
        # Vector 0 at time=0.0: no rotation, no translation -> [1,0,0]
        # Vector 1 at time=0.5: 45° rotation, half translation
        # Vector 2 at time=1.0: 90° rotation, full translation
        expected = np.array([
            [1., 0., 0.],  # time=0
            [np.cos(np.pi/4) + 0.5, np.sin(np.pi/4) + 1.0, 0.0 + 1.5],  # time=0.5
            [0. + 1., 1. + 2., 0. + 3.]  # time=1.0, [1,0,0] rotated 90° = [0,1,0]
        ])
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5, atol=1e-10)

    def test_rotate_only_scalar_time(self):
        """Test rotate method with scalar time."""
        vectors = np.array([
            [1., 0., 0.],
            [0., 1., 0.]
        ])
        time = 1.0
        
        result = self.motion.rotate(vectors, time)
        
        # Should be shape (2, 3)
        self.assertEqual(result.shape, (2, 3))
        
        # Rotate 90° around z-axis (no translation)
        # [1,0,0] -> [0,1,0]
        # [0,1,0] -> [-1,0,0]
        expected = np.array([
            [0., 1., 0.],
            [-1., 0., 0.]
        ])
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5, atol=1e-10)

    def test_rotate_only_vector_times(self):
        """Test rotate method with per-vector times."""
        vectors = np.array([
            [1., 0., 0.],
            [1., 0., 0.],
            [1., 0., 0.]
        ])
        times = np.array([0.0, 0.5, 1.0])
        
        result = self.motion.rotate(vectors, times)
        
        # Should be shape (3, 3)
        self.assertEqual(result.shape, (3, 3))
        
        # Each vector rotated by different angle
        expected = np.array([
            [1., 0., 0.],  # 0° rotation
            [np.cos(np.pi/4), np.sin(np.pi/4), 0.],  # 45° rotation
            [0., 1., 0.]  # 90° rotation
        ])
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5, atol=1e-10)

    def test_different_vectors_different_times(self):
        """Test with different vectors and different times - comprehensive test."""
        vectors = np.array([
            [2., 0., 0.],
            [0., 3., 0.],
            [1., 1., 0.],
            [0., 0., 1.]
        ])
        times = np.array([0.0, 0.25, 0.5, 1.0])
        
        result = self.motion(vectors, times)
        
        # Should be shape (4, 3)
        self.assertEqual(result.shape, (4, 3))
        
        # Manually compute each expected result
        expected = []
        for i, (vec, t) in enumerate(zip(vectors, times)):
            angle = self.rotation_angle * t
            # Rotation around z-axis
            rot_vec = np.array([
                vec[0] * np.cos(angle) - vec[1] * np.sin(angle),
                vec[0] * np.sin(angle) + vec[1] * np.cos(angle),
                vec[2]
            ])
            # Add translation
            trans_vec = rot_vec + self.translation * t
            expected.append(trans_vec)
        
        expected = np.array(expected)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5, atol=1e-10)

    def test_consistency_scalar_vs_vector(self):
        """Test that scalar time gives same result as broadcasting to vector times."""
        vectors = np.array([
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.]
        ])
        time_scalar = 0.7
        time_vector = np.array([0.7, 0.7, 0.7])
        
        result_scalar = self.motion(vectors, time_scalar)
        result_vector = self.motion(vectors, time_vector)
        
        # Should be identical
        np.testing.assert_allclose(result_scalar.numpy(), result_vector.numpy(), rtol=1e-10)

    def test_motion_with_origin(self):
        """Test motion with non-zero origin."""
        origin = np.array([1., 1., 0.])
        motion = RigidBodyMotion(
            self.rotation_axis,
            self.rotation_angle,
            self.translation,
            origin
        )
        
        vectors = np.array([
            [2., 1., 0.],  # Point at (2,1,0), offset (1,0,0) from origin
            [1., 2., 0.]   # Point at (1,2,0), offset (0,1,0) from origin
        ])
        times = np.array([1.0, 1.0])
        
        result = motion(vectors, times)
        
        # Should be shape (2, 3)
        self.assertEqual(result.shape, (2, 3))
        
        # Point 1: offset [1,0,0] from origin, rotate 90° -> [0,1,0], add back origin [1,1,0] = [1,2,0], translate [1,2,3] = [2,4,3]
        # Point 2: offset [0,1,0] from origin, rotate 90° -> [-1,0,0], add back origin [1,1,0] = [0,1,0], translate [1,2,3] = [1,3,3]
        expected = np.array([
            [2., 4., 3.],
            [1., 3., 3.]  # Fixed: [0,1,0] + [1,2,3] = [1,3,3]
        ])
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5, atol=1e-10)

    def test_edge_case_single_vector_in_batch(self):
        """Test edge case with batch of 1 vector."""
        vectors = np.array([[1., 0., 0.]])  # Shape (1, 3)
        times = np.array([0.5])  # Shape (1,)
        
        result = self.motion(vectors, times)
        
        # Should be shape (1, 3)
        self.assertEqual(result.shape, (1, 3))

    def test_large_batch(self):
        """Test with large batch to ensure performance."""
        N = 10000
        vectors = np.random.randn(N, 3)
        times = np.random.rand(N)
        
        result = self.motion(vectors, times)
        
        # Should be shape (N, 3)
        self.assertEqual(result.shape, (N, 3))
        
        # Verify a few random samples manually
        for _ in range(5):
            idx = np.random.randint(0, N)
            vec = vectors[idx]
            t = times[idx]
            
            # Compute expected manually
            angle = self.rotation_angle * t
            rot_vec = np.array([
                vec[0] * np.cos(angle) - vec[1] * np.sin(angle),
                vec[0] * np.sin(angle) + vec[1] * np.cos(angle),
                vec[2]
            ])
            expected_single = rot_vec + self.translation * t
            
            np.testing.assert_allclose(result[idx].numpy(), expected_single, rtol=1e-5, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
