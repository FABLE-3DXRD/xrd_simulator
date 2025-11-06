# Cylindrical Mesh Generation with meshpy

The `run_test.py` script now uses **meshpy** to generate cylindrical tetrahedral meshes for XRD simulations.

## Quick Reference

### Parameters

```python
sample_params = {
    "cylinder_radius": 100.0,           # microns - radius of cylinder
    "cylinder_height": 200.0,           # microns - height of cylinder
    "max_cell_circumradius": 30.0,      # microns - controls mesh density
    "phase": "Fe_BCC",
}
```

### Mesh Density Control

The `max_cell_circumradius` parameter controls the mesh density:

| max_cell_circumradius | Approximate # Elements | Use Case |
|-----------------------|------------------------|----------|
| 50 µm | ~100-500 | Quick tests, prototyping |
| 30 µm | ~500-2000 | Standard simulations |
| 20 µm | ~2000-5000 | High-resolution simulations |
| 10 µm | ~10000+ | Very fine simulations (slow) |

**Rule of thumb**: Smaller values = more tetrahedral elements = finer mesh = slower simulation

## How It Works

The `generate_cylinder_mesh()` function:

1. **Creates boundary points**: Generates circular points at top and bottom of cylinder
2. **Defines facets**: Specifies the boundary surfaces (top circle, bottom circle, sides)
3. **Meshes interior**: Uses meshpy.tet to fill the cylinder with tetrahedra
4. **Volume constraint**: Controls element size via `max_volume = (max_cell_circumradius^3) / 6`

## Example Usage

### Standard Cylinder
```python
mesh = generate_cylinder_mesh(
    radius=100.0,          # 100 µm radius
    height=200.0,          # 200 µm height
    max_cell_circumradius=30.0   # ~1000-2000 tets
)
```

### Fine Mesh
```python
mesh = generate_cylinder_mesh(
    radius=100.0,
    height=200.0,
    max_cell_circumradius=15.0   # ~5000-10000 tets
)
```

### Coarse Mesh (Fast)
```python
mesh = generate_cylinder_mesh(
    radius=100.0,
    height=200.0,
    max_cell_circumradius=50.0   # ~200-500 tets
)
```

## Advantages of meshpy Over pygalmesh

1. **Python 3.13 Support**: meshpy supports newer Python versions
2. **Lighter Dependencies**: Fewer external C++ library requirements
3. **Explicit Control**: Direct control over mesh generation parameters
4. **Fast**: Efficient tetrahedral mesh generation

## Mesh Output

The generated mesh is saved as:
- **Binary format**: `.pc` file (dill pickle)
- **Visualization format**: `.xdmf` file (for ParaView, VisIt, etc.)

Filename includes dimensions for easy identification:
```
cylinder_R100.0_H200.0.pc
cylinder_R100.0_H200.0.xdmf
```

## Visualizing the Mesh

Load the `.xdmf` file in ParaView:
1. Open ParaView
2. File → Open → Select the `.xdmf` file
3. Click "Apply" in the Properties panel
4. Visualize the mesh colored by orientation, element index, etc.

## Integration with XRD Simulation

The cylindrical mesh integrates seamlessly with the rest of the XRD simulation pipeline:

```python
# 1. Generate mesh
mesh = generate_cylinder_mesh(radius=100, height=200, max_cell_circumradius=30)

# 2. Assign random orientations
orientation = R.random(mesh.number_of_elements).as_matrix()

# 3. Create polycrystal
polycrystal = Polycrystal(mesh, orientation, strain=np.zeros((3,3)), 
                          phases=phases, element_phase_map=element_phase_map)

# 4. Run diffraction simulation
peaks_dict = polycrystal.diffract(beam, detector, motion)
pattern = detector.render(peaks_dict, frames_to_render=0, method='gauss')
```

## Performance Considerations

- **Memory**: Larger meshes require more RAM
- **GPU**: GPU acceleration helps with rendering, not mesh generation
- **Time**: Diffraction time scales roughly linearly with number of elements

**Benchmark** (approximate, depends on hardware):
- 500 elements: ~1-5 seconds
- 2000 elements: ~5-20 seconds  
- 10000 elements: ~30-120 seconds

## See Also

- Main script: `scripts/run_test.py`
- Mesh module: `xrd_simulator/mesh.py`
- Templates module: `xrd_simulator/templates.py` (has `polycrystal_from_odf` with cylinder generation)
