"""Field.gather_data against assembling every local block with np.block (the original implementation)."""

import pytest
import numpy as np
from math import prod
from mpi4py import MPI
import dedalus.public as d3

comm = MPI.COMM_WORLD


def gather_reference(field, root=0):
    pieces = field.dist.comm.gather(field.data, root=root)
    if field.dist.comm.rank == root:
        ext_mesh = field.layout.ext_mesh
        combined = np.zeros(prod(ext_mesh), dtype=object)
        combined[:] = pieces
        return np.block(combined.reshape(ext_mesh).tolist())


def mesh_2d():
    for m0 in range(int(np.sqrt(comm.size)), 0, -1):
        if comm.size % m0 == 0:
            return (comm.size // m0, m0)


def shell_fields():
    coords = d3.SphericalCoordinates('phi', 'theta', 'r')
    dist = d3.Distributor(coords, dtype=np.float64, mesh=mesh_2d())
    shell = d3.ShellBasis(coords, shape=(16, 8, 10), radii=(0.5, 1.5), dealias=3/2, dtype=np.float64)
    T = dist.Field(name='T', bases=shell)
    v = dist.VectorField(coords, name='v', bases=shell)
    T.fill_random('g', seed=1)
    v.fill_random('g', seed=2)
    integral = d3.integ(T*T).evaluate()
    profile = d3.Average(v@v, coords.S2coordsys).evaluate()
    return [T, v, integral, profile]


def box_fields():
    coords = d3.CartesianCoordinates('x', 'y', 'z')
    dist = d3.Distributor(coords, dtype=np.float64, mesh=mesh_2d())
    xb = d3.RealFourier(coords['x'], size=12, bounds=(0, 1), dealias=3/2)
    yb = d3.RealFourier(coords['y'], size=8, bounds=(0, 1), dealias=3/2)
    zb = d3.Chebyshev(coords['z'], size=10, bounds=(0, 1), dealias=3/2)
    u = dist.VectorField(coords, name='u', bases=(xb, yb, zb))
    u.fill_random('g', seed=3)
    c = dist.Field(name='c', bases=zb)
    c.fill_random('g', seed=4)
    return [u, c, d3.integ(u@u).evaluate()]


@pytest.mark.parametrize('build', [shell_fields, box_fields])
@pytest.mark.parametrize('layout', ['g', 'c'])
@pytest.mark.parametrize('scales', [1, 3/2])
def test_gather_matches_block(build, layout, scales):
    for field in build():
        field.change_scales(scales)
        field.change_layout(layout)
        ref = gather_reference(field)
        out = field.gather_data()
        if comm.rank == 0:
            assert out.shape == ref.shape
            assert out.dtype == ref.dtype
            assert np.array_equal(out, ref)
        else:
            assert out is None
