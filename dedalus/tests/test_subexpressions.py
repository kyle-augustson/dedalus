"""Common-subexpression elimination (core/subexpressions.py): merging and bitwise-identical results."""

import pytest
import numpy as np
import dedalus.public as d3
from dedalus.core import subexpressions
from dedalus.core.evaluator import Evaluator


class cse_enabled:
    def __init__(self, enabled):
        self.enabled = enabled
    def __enter__(self):
        self.saved = subexpressions.ENABLED
        subexpressions.ENABLED = self.enabled
    def __exit__(self, *args):
        subexpressions.ENABLED = self.saved


def burgers(N, nsteps):
    xcoord = d3.Coordinate('x')
    dist = d3.Distributor(xcoord, dtype=np.float64)
    xbasis = d3.RealFourier(xcoord, size=N, bounds=(0, 2*np.pi), dealias=3/2)
    x = dist.local_grid(xbasis)
    u = dist.Field(name='u', bases=xbasis)
    dx = lambda A: d3.Differentiate(A, xcoord)
    u['g'] = np.sin(x) + 0.3*np.cos(3*x)
    # dx(u) appears three times on the RHS, twice from the string and once from the Python expression
    extra = 0.1*dx(u)*u
    problem = d3.IVP([u], namespace=locals())
    problem.add_equation("dt(u) - 0.05*dx(dx(u)) = -u*dx(u) - 0.1*dx(u)*dx(u) + extra")
    solver = problem.build_solver(d3.SBDF2)
    for i in range(nsteps):
        solver.step(1e-3)
    return u['c'].copy(), solver


@pytest.mark.parametrize('N', [32, 64])
def test_ivp_bitwise(N):
    with cse_enabled(False):
        ref, _ = burgers(N, 20)
    with cse_enabled(True):
        out, solver = burgers(N, 20)
    assert np.array_equal(ref, out)


def tasks_2d(dist, coords, xbasis, zbasis):
    u = dist.Field(name='u', bases=(xbasis, zbasis))
    x, z = dist.local_grids(xbasis, zbasis)
    u['g'] = np.sin(x) * z**2 + np.cos(2*x) * z
    dx = lambda A: d3.Differentiate(A, coords['x'])
    dz = lambda A: d3.Differentiate(A, coords['z'])
    t1 = dx(u)*dx(u) + dz(u)
    t2 = dx(u)*u + dz(u)*u + np.exp(-u)
    t3 = d3.grad(u)@d3.grad(u) + np.exp(-u)
    return [t1, t2, t3]


def evaluate_2d(enabled):
    coords = d3.CartesianCoordinates('x', 'z')
    dist = d3.Distributor(coords, dtype=np.float64)
    xbasis = d3.RealFourier(coords['x'], size=16, bounds=(0, 2*np.pi), dealias=3/2)
    zbasis = d3.Chebyshev(coords['z'], size=12, bounds=(0, 1), dealias=3/2)
    evaluator = Evaluator(dist, {})
    handler = evaluator.add_dictionary_handler(iter=1)
    for i, task in enumerate(tasks_2d(dist, coords, xbasis, zbasis)):
        handler.add_task(task, name='t%i' % i)
    with cse_enabled(enabled):
        evaluator.evaluate_handlers([handler])
    return [handler.fields['t%i' % i]['g'].copy() for i in range(3)]


def test_merge_2d():
    ref = evaluate_2d(False)
    out = evaluate_2d(True)
    for a, b in zip(ref, out):
        assert np.array_equal(a, b)


def test_merge_count_2d():
    coords = d3.CartesianCoordinates('x', 'z')
    dist = d3.Distributor(coords, dtype=np.float64)
    xbasis = d3.RealFourier(coords['x'], size=16, bounds=(0, 2*np.pi), dealias=3/2)
    zbasis = d3.Chebyshev(coords['z'], size=12, bounds=(0, 1), dealias=3/2)
    evaluator = Evaluator(dist, {})
    handler = evaluator.add_dictionary_handler(iter=1)
    for i, task in enumerate(tasks_2d(dist, coords, xbasis, zbasis)):
        handler.add_task(task, name='t%i' % i)
    # dx(u) x2 and dz(u) x1 merge into earlier copies, exp(-u) x1 (with its -1*u) and grad(u) x1
    assert subexpressions.eliminate(evaluator.handlers) >= 5
    # A second pass finds nothing new
    assert subexpressions.eliminate(evaluator.handlers) == 0


def test_general_function_not_merged():
    coords = d3.CartesianCoordinates('x', 'z')
    dist = d3.Distributor(coords, dtype=np.float64)
    xbasis = d3.RealFourier(coords['x'], size=8, bounds=(0, 2*np.pi))
    zbasis = d3.Chebyshev(coords['z'], size=8, bounds=(0, 1))
    calls = []
    def func(*args):
        calls.append(1)
        return np.ones(tuple(dist.grid_layout.local_shape(u.domain, 1)))
    u = dist.Field(name='u', bases=(xbasis, zbasis))
    u['g'] = 1
    gf1 = d3.GeneralFunction(dist, u.domain, (), np.float64, 'g', func, args=[])
    gf2 = d3.GeneralFunction(dist, u.domain, (), np.float64, 'g', func, args=[])
    evaluator = Evaluator(dist, {})
    handler = evaluator.add_dictionary_handler(iter=1)
    handler.add_task(gf1 + u, name='a')
    handler.add_task(gf2 + u, name='b')
    with cse_enabled(True):
        evaluator.evaluate_handlers([handler])
    assert len(calls) == 2


def evaluate_shell(enabled):
    coords = d3.SphericalCoordinates('phi', 'theta', 'r')
    dist = d3.Distributor(coords, dtype=np.float64)
    shell = d3.ShellBasis(coords, shape=(16, 8, 10), radii=(0.5, 1.5), dealias=3/2, dtype=np.float64)
    T = dist.Field(name='T', bases=shell)
    v = dist.VectorField(coords, name='v', bases=shell)
    T.fill_random('c', seed=2)
    v.fill_random('c', seed=3)
    T.low_pass_filter(scales=0.5)
    v.low_pass_filter(scales=0.5)
    strain = d3.grad(v) + d3.transpose(d3.grad(v))
    tasks = [-v@d3.grad(T) - d3.grad(T)@d3.grad(T),
             -v@d3.grad(v) + d3.grad(T)*T + d3.div(strain),
             d3.trace(strain@d3.grad(v)) + np.exp(-T)*d3.div(d3.grad(T))]
    evaluator = Evaluator(dist, {})
    handler = evaluator.add_dictionary_handler(iter=1)
    for i, task in enumerate(tasks):
        handler.add_task(task, name='t%i' % i)
    with cse_enabled(enabled):
        evaluator.evaluate_handlers([handler])
    return [handler.fields['t%i' % i]['c'].copy() for i in range(len(tasks))]


def test_merge_shell():
    ref = evaluate_shell(False)
    out = evaluate_shell(True)
    for a, b in zip(ref, out):
        assert np.array_equal(a, b)
