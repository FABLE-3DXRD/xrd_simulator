import numpy as np
import pygalmesh
import time
import copy
from pyevtk.hl import gridToVTK
from numba import njit


def save_as_vtk_voxel_volume(
    file, voxel_volume, name, units=(
        1., 1., 1.), origin=(
            0., 0., 0.)):
    """Save numpy array with voxel information to paraview readable format.
    Args:
        file (:obj:`string`): Absolute path ending with desired filename.
        voxel_volume (:obj:`numpy array`): Per voxel density values in a 3d array.
        units (:obj:`tuple` of :obj:`float`): Distance between voxels in ```voxel_volume``` (x,y,z).
        origin (:obj:`tuple` of :obj:`float`): Origin of voxel ```voxel_volume[0,0,0]``` (x,y,z).
    """
    x = np.arange(
        0,
        voxel_volume.shape[0] + 1,
        dtype=np.float32) * units[0] + origin[0]
    y = np.arange(
        0,
        voxel_volume.shape[1] + 1,
        dtype=np.float32) * units[1] + origin[1]
    z = np.arange(
        0,
        voxel_volume.shape[2] + 1,
        dtype=np.float32) * units[2] + origin[2]
    gridToVTK(file, x, y, z, cellData={name: voxel_volume})


@njit
def label_volume(X, Y, Z, R, seed_labels, seed_coord):
    labeled_volume = np.ones(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                if X[i, j, k] * X[i, j, k] + Y[i, j, k] * Y[i, j, k] < R * R:
                    x = np.array([X[i, j, k], Y[i, j, k], Z[i, j, k]])
                    a = seed_coord - x
                    labeled_volume[i, j, k] = seed_labels[np.argmin(
                        np.sum(a * a, axis=1))]
    return labeled_volume


t1 = time.process_time()
g = np.linspace(-0.5, 0.5, 256)
X, Y, Z = np.meshgrid(g, g, g, indexing='ij')
number_of_desired_seed_points = 64
seed_coord = np.random.rand(number_of_desired_seed_points, 3) - 0.5
seed_labels = np.array(range(2, seed_coord.shape[0] + 2)).astype(np.uint16)
R = 0.5
labeled_volume = label_volume(X, Y, Z, R, seed_labels, seed_coord)

grain_labels = np.unique(labeled_volume)
grain_labels = grain_labels[grain_labels != 1]

new_labels = np.linspace(
    2,
    1 + len(grain_labels),
    len(grain_labels)).astype(
        np.uint16)
print(new_labels)

new_labeled_volume = np.ones(labeled_volume.shape)
for gl, nl in zip(grain_labels, new_labels):
    new_labeled_volume[labeled_volume == gl] = nl
labeled_volume = new_labeled_volume

t2 = time.process_time()
print("Tesselation time: ", t2 - t1)

save_as_vtk_voxel_volume(
    "voroni_sample/voxelated_voroni_sample",
    labeled_volume,
    name='Labeled Grains',
    units=(
        1. / len(g),
        1. / len(g),
        1. / len(g)),
    origin=(
        0.,
        0.,
        0.))

dx = 0.03
for label in new_labels:
    grain = labeled_volume.copy()
    grain[grain != label] = 0
    if np.sum(grain) != 0:
        mesh = pygalmesh.generate_from_array(
            grain.astype(
                np.uint16), [
                1. / len(g)] * 3, max_facet_distance=dx, max_cell_circumradius=dx, verbose=False)
        mesh.cell_data['label'] = copy.copy(mesh.cell_data['medit:ref'])
        mesh.cell_data['label'][0] = (
            np.ones(
                mesh.cell_data['label'][0].shape) *
            label).astype(
            np.uint16)
        mesh.cell_data['label'][1] = (
            np.ones(
                mesh.cell_data['label'][1].shape) *
            label).astype(
            np.uint16)
        mesh.write("grain_meshes/grain" + str(label).zfill(4) + ".xdmf")
        print("Wrote " +
              "voroni_sample/grain_meshes/grain" +
              str(label).zfill(4) +
              ".xdmf to file with " +
              str(mesh.cells_dict['tetra'].shape[0]) +
              " elements in the mesh")
