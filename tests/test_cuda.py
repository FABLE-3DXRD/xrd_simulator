"""Tests for CUDA/device configuration module."""

import unittest
import torch
from xrd_simulator.cuda import configure_device, get_selected_device


class TestCuda(unittest.TestCase):
    """Tests for device configuration functions."""

    def setUp(self):
        """Store original device state."""
        self.original_device = get_selected_device()

    def tearDown(self):
        """Restore original device state."""
        configure_device(self.original_device, verbose=False)

    def test_configure_device_cpu(self):
        """Test forcing CPU device."""
        result = configure_device("cpu", verbose=False)
        self.assertEqual(result, "cpu")
        self.assertEqual(get_selected_device(), "cpu")

    def test_configure_device_auto(self):
        """Test auto device selection."""
        result = configure_device("auto", verbose=False)
        # Should return either 'cpu' or 'cuda'
        self.assertIn(result, ["cpu", "cuda"])
        self.assertEqual(get_selected_device(), result)

    def test_configure_device_none(self):
        """Test None device selection (same as auto)."""
        result = configure_device(None, verbose=False)
        self.assertIn(result, ["cpu", "cuda"])
        self.assertEqual(get_selected_device(), result)

    def test_configure_device_gpu_alias(self):
        """Test 'gpu' as alias for 'cuda'."""
        result = configure_device("gpu", verbose=False)
        # Returns 'cuda' if available, otherwise 'cpu'
        self.assertIn(result, ["cpu", "cuda"])

    def test_configure_device_invalid(self):
        """Test invalid device raises ValueError."""
        with self.assertRaises(ValueError):
            configure_device("invalid_device", verbose=False)

    def test_get_selected_device(self):
        """Test get_selected_device returns configured device."""
        configure_device("cpu", verbose=False)
        self.assertEqual(get_selected_device(), "cpu")

    def test_configure_device_sets_torch_default(self):
        """Test that configure_device sets torch default device."""
        configure_device("cpu", verbose=False)
        # Create a tensor and check its device
        t = torch.zeros(1)
        self.assertEqual(str(t.device), "cpu")


if __name__ == "__main__":
    unittest.main()
