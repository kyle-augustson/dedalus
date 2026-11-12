"""GROUP_TRANSPOSES = true, false and auto (with thresholds on both sides of the rank count) give identical results."""

import pytest
import numpy as np
from mpi4py import MPI
import dedalus.public as d3
from dedalus.core import distributor
from dedalus.core.evaluator import Evaluator

comm = MPI.COMM_WORLD


def mesh_2d():
    for m0 in range(int(np.sqrt(comm.size)), 0, -1):
        if comm.size % m0 == 0:
            return (comm.size // m0, m0)


def shell_tasks():
    coords = d3.SphericalCoordinates('phi', 'theta', 'r')
    dist = d3.Distributor(coords, dtype=np.float64, mesh=mesh_2d())
    shell = d3.ShellBasis(coords, shape=(16, 8, 10), radii=(0.5, 1.5), dealias=3/2, dtype=np.float64)
    T = dist.Field(name='T', bases=shell)
    v = dist.VectorField(coords, name='v', bases=shell)
    T.fill_random('g', seed=1)
    v.fill_random('g', seed=2)
    T.low_pass_filter(scales=0.5)
    v.low_pass_filter(scales=0.5)
    # Evaluate both tasks through one handler, so several fields are transposed together
    tasks = [-v@d3.grad(v) + T*d3.grad(T), v@d3.grad(T) + T*T]
    evaluator = Evaluator(dist, {})
    handler = evaluator.add_dictionary_handler(iter=1)
    for i, task in enumerate(tasks):
        handler.add_task(task, name='t%i' % i)
    evaluator.evaluate_handlers([handler])
    return [handler.fields['t%i' % i] for i in range(len(tasks))]


@pytest.mark.parametrize('setting, threshold, grouped', [
    (True, 2048, True),
    (False, 2048, False),
    ('auto', comm.size, False),
    ('auto', comm.size - 1, True),
    ('AUTO ', 2048, False),
])
def test_group_transposes(setting, threshold, grouped, monkeypatch):
    monkeypatch.setattr(distributor, 'GROUP_TRANSPOSES', setting)
    monkeypatch.setattr(distributor, 'GROUP_TRANSPOSES_RANKS', threshold)
    assert distributor.group_transposes(comm.size) == grouped
    calls = {'group': 0}
    for name in ('increment_group', 'decrement_group'):
        original = getattr(distributor.Transpose, name)
        def counted(self, fields, original=original):
            calls['group'] += 1
            return original(self, fields)
        monkeypatch.setattr(distributor.Transpose, name, counted)
    out = [task['c'].copy() for task in shell_tasks()]
    monkeypatch.setattr(distributor, 'GROUP_TRANSPOSES', False)
    ref = [task['c'].copy() for task in shell_tasks()]
    for a, b in zip(out, ref):
        assert np.array_equal(a, b)
    if comm.size > 1:
        assert (calls['group'] > 0) == grouped


@pytest.mark.parametrize('value, grouped', [('true', True), ('off', False), ('Yes', True), ('0', False)])
def test_group_transposes_strings(value, grouped, monkeypatch):
    monkeypatch.setattr(distributor, 'GROUP_TRANSPOSES', value)
    assert distributor.group_transposes(comm.size) == grouped


def test_group_transposes_invalid(monkeypatch):
    monkeypatch.setattr(distributor, 'GROUP_TRANSPOSES', 'sometimes')
    with pytest.raises(ValueError):
        distributor.group_transposes(comm.size)
