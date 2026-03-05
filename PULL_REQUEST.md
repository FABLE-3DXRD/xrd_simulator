# Powder Diffraction Simulation with PyTorch Acceleration

## Summary

This PR introduces a major overhaul of `xrd_simulator` to support **powder diffraction simulation** with **GPU-accelerated rendering** via PyTorch. The refactor enables realistic simulation of polycrystalline samples across all grain size regimes—from nanocrystals to large single crystals—with physically accurate peak broadening models.

## Key Features

### 🚀 PyTorch Backend
- Complete migration from NumPy to PyTorch for all core computations
- Automatic CPU/GPU device selection via new `xrd_simulator.cuda` module
- Memory-efficient batch processing for millions of grains

### 🔬 Multi-Scale Rendering Methods
New rendering system with three physically-motivated methods:

| Method | Grain Size | Description |
|--------|------------|-------------|
| `nano` | < 0.1³ µm³ | Airy disk patterns with Scherrer broadening for nanocrystals |
| `micro` | 0.1³ - pixel³ µm³ | Fast Gaussian profiles for powder patterns |
| `macro` | > pixel³ µm³ | 3D volume projection showing crystal morphology |
| `auto` | All sizes | Automatic method selection based on grain volume |

### 📊 Physical Corrections
- **Lorentz factor**: Proper intensity corrections for diffraction geometry
- **Polarization factor**: X-ray polarization effects
- **Structure factors**: Crystallographic intensity modulation
- New `scattering_factors.py` module with Scherrer formula implementation

### ⚡ Performance Improvements
- Batch rendering of 1M+ grains in seconds on GPU
- Intelligent memory management with automatic batching
- Removed NumPy dependency from hot paths

## Breaking Changes

- `ScatteringUnit` class removed (functionality merged into `Polycrystal` and `Detector`)
- Rendering method names changed: `centroid` → `micro`, `profiles` → `nano`, `volumes` → `macro`
- Minimum Python version: 3.10
- PyTorch 2.5+ required

## Testing

- ✅ **82/82 tests passing**
- New end-to-end tests for powder diffraction with pyFAI peak finding
- New crystallite size broadening validation tests
- CUDA device configuration tests

## Files Changed

- **100 files** modified (+10,139 / -5,764 lines)
- Core modules refactored: `detector.py`, `polycrystal.py`, `motion.py`, `mesh.py`, `laue.py`
- New modules: `cuda.py`, `scattering_factors.py`
- Documentation rebuilt with Sphinx

## Dependencies Updated

```
torch>=2.5.0
numpy>=1.24.0
scipy>=1.11.0
pandas>=2.0.0
```

## Commits (20)

1. Vectorization merge and initial refactoring
2. PyTorch backend implementation through `_diffract`
3. Torch/NumPy equivalence achieved for full simulation pipeline
4. Multi-frame rendering support
5. Powder diffraction mode with PSF convolution
6. Lorentz, polarization, and structure factor corrections
7. Memory-efficient batch processing
8. Removed NumPy from computations
9. Volume projection for macro grains
10. Gaussian interpolation for micro grains
11. Airy disk patterns for nano grains
12. Auto mode with grain-size-based method selection
13. Vectorized `RigidBodyMotion` for batch operations
14. End-to-end powder diffraction tests
15. CUDA device configuration module
16. Scherrer formula implementation
17. Documentation and docstring updates
18. CI/CD workflow updates for Python 3.13
19. Repository cleanup
20. Sphinx documentation rebuild

---

**Ready for review and merge into `main`.**
