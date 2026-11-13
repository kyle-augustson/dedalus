"""Test file output in parallel: gather, virtual and mpio file handlers, and merging virtual output."""

import pathlib
import shutil
import tempfile
import pytest
import numpy as np
import h5py
from mpi4py import MPI
import dedalus.public as d3
from dedalus.tools import post
from dedalus.tools.cache import CachedFunction

comm = MPI.COMM_WORLD


def mesh_2d():
    for m0 in range(int(np.sqrt(comm.size)), 0, -1):
        if comm.size % m0 == 0:
            return (comm.size // m0, m0)


class shared_tempdir:
    """Temporary directory created by rank 0 and shared by all ranks."""
    def __enter__(self):
        path = tempfile.mkdtemp(dir='.') if comm.rank == 0 else None
        self.path = comm.bcast(path, root=0)
        return self.path
    def __exit__(self, *args):
        comm.Barrier()
        if comm.rank == 0:
            shutil.rmtree(self.path)


@CachedFunction
def build_domain(N, dealias, dtype):
    # Cached so the test cases share distributors: each distributor holds MPI communicators that are never freed
    # (its Cartesian communicator, one per transpose, and those of the FFTW-MPI plans), and a new one per case
    # exhausts the MPI library's communicator limit (4096 per process with MPICH) on some ranks.
    c = d3.CartesianCoordinates('x', 'y', 'z')
    d = d3.Distributor(c, dtype=dtype, mesh=mesh_2d())
    xb = d3.Fourier(c.coords[0], size=N, bounds=(0, 1), dealias=dealias, dtype=dtype)
    yb = d3.Fourier(c.coords[1], size=N, bounds=(0, 1), dealias=dealias, dtype=dtype)
    zb = d3.Chebyshev(c.coords[2], size=N, bounds=(0, 1), dealias=dealias)
    return c, d, xb, yb, zb


def build_cartesian(N, dealias, dtype):
    c, d, xb, yb, zb = build_domain(N, dealias, dtype)
    x, y, z = d.local_grids(xb, yb, zb)
    u = d.Field(name='u', bases=(xb, yb, zb))
    v = d.VectorField(c, name='v', bases=(xb, yb, zb))
    u['g'] = np.sin(2*np.pi*x) * np.sin(2*np.pi*y) * np.sin(z)
    v['g'][0] = np.cos(2*np.pi*x) * z
    v['g'][1] = np.sin(2*np.pi*y) * z**2
    v['g'][2] = np.cos(2*np.pi*(x + y))
    tasks = [u, u(x=0), u(y=0), u(z=0), u(x=0, y=0), u(x=0, z=0), u(y=0, z=0), u(x=0, y=0, z=0),
             v, v(x=0), v(z=0), d3.integ(u), d3.Average(u, c.coords[0])]
    return d, u, v, tasks


def check_output(d, tasks, path, layout, scales):
    """
    Compare each rank's local data with the saved global data (last write). Mismatches are gathered from all ranks
    before asserting, so a failure on some ranks fails the test on every rank instead of leaving the others waiting.
    """
    # Evaluate on all ranks (collective), then compare locally
    local = []
    for task in tasks:
        name = str(task)
        # A copy, so changing its scales does not truncate a state field that later tasks are evaluated from
        task = task.evaluate().copy()
        task.change_scales(scales)
        data = task[layout]
        local.append((name, data, (slice(None),) * len(task.tensorsig) + task.layout.slices(task.domain, task.scales)))
    bad = []
    try:
        with h5py.File(path, mode='r') as file:
            for name, data, local_slices in local:
                if not data.size:
                    continue
                saved = file['tasks'][name][-1][local_slices]
                if saved.shape != data.shape:
                    bad.append(f'rank {comm.rank} {name}: shape {saved.shape} saved, {data.shape} local')
                elif not np.allclose(data, saved):
                    bad.append(f'rank {comm.rank} {name}: max error {np.max(np.abs(data - saved)):.2e}')
    except Exception as e:
        bad.append(f'rank {comm.rank}: {type(e).__name__}: {e}')
    bad = sum(comm.allgather(bad), [])
    assert not bad, '; '.join(bad[:6])


@pytest.mark.parametrize('parallel', ['gather', 'virtual', 'mpio'])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('dealias', [1, 3/2])
@pytest.mark.parametrize('output_scales', [1/2, 1, 3/2])
@pytest.mark.parametrize('output_layout', ['g', 'c'])
def test_cartesian_output(parallel, dtype, dealias, output_scales, output_layout):
    d, u, v, tasks = build_cartesian(8, dealias, dtype)
    problem = d3.IVP([u, v])
    problem.add_equation("dt(u) = 0")
    problem.add_equation("dt(v) = 0")
    solver = problem.build_solver("RK111")
    with shared_tempdir() as tempdir:
        output = solver.evaluator.add_file_handler(tempdir, iter=1, max_writes=1, parallel=parallel)
        for task in tasks:
            output.add_task(task, layout=output_layout, name=str(task), scales=output_scales)
        solver.evaluate_handlers([output])
        comm.Barrier()
        check_output(d, tasks, f'{tempdir}/{pathlib.Path(tempdir).stem}_s1.h5', output_layout, output_scales)


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('output_scales', [1/2, 1])
def test_cartesian_output_merged_virtual(dtype, output_scales):
    d, u, v, tasks = build_cartesian(8, 3/2, dtype)
    problem = d3.IVP([u, v])
    problem.add_equation("dt(u) = 0")
    problem.add_equation("dt(v) = 0")
    solver = problem.build_solver("RK111")
    with shared_tempdir() as tempdir:
        output = solver.evaluator.add_file_handler(tempdir, iter=1, max_writes=1, parallel='virtual')
        for task in tasks:
            output.add_task(task, layout='g', name=str(task), scales=output_scales)
        solver.evaluate_handlers([output])
        comm.Barrier()
        post.merge_virtual_analysis(tempdir, cleanup=True)
        comm.Barrier()
        path = f'{tempdir}/{pathlib.Path(tempdir).stem}_s1.h5'
        with h5py.File(path, mode='r') as file:
            virtual = [name for name in file['tasks'] if file['tasks'][name].is_virtual]
        assert not virtual, virtual
        check_output(d, tasks, path, 'g', output_scales)
