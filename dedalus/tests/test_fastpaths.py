"""Fast paths (dedalus.tools.fastpaths) against the original code paths: operators, transforms, products, IVP steps."""

import pytest
import numpy as np
import dedalus.public as d3
from dedalus.tools import fastpaths


pytestmark = pytest.mark.skipif(fastpaths._kern is None, reason="fast_kernels extension not built")


def _basis(case, coords, dealias):
    if case == 'shell':
        return d3.ShellBasis(coords, shape=(32, 16, 12), radii=(0.5, 1.5), dealias=dealias, dtype=np.float64)
    if case == 'ball':
        return d3.BallBasis(coords, shape=(32, 16, 12), radius=1, dealias=dealias, dtype=np.float64)
    return d3.SphereBasis(coords.S2coordsys, shape=(32, 16), radius=1, dealias=dealias, dtype=np.float64)


EXPRESSIONS = {
    'T': lambda T, u: T + 0, 'u': lambda T, u: u + 0,
    'grad u': lambda T, u: d3.grad(u), 'curl u': lambda T, u: d3.curl(u), 'div u': lambda T, u: d3.div(u),
    'lap T': lambda T, u: d3.lap(T), 'grad grad T': lambda T, u: d3.grad(d3.grad(T)),
    'u x curl u': lambda T, u: d3.cross(u, d3.curl(u)), 'u . grad T': lambda T, u: u @ d3.grad(T),
    'curl u + u': lambda T, u: d3.curl(u) + u,
}


def _evaluate(case, dealias, enabled):
    old = fastpaths.ENABLED
    fastpaths.ENABLED = enabled
    try:
        coords = d3.SphericalCoordinates('phi', 'theta', 'r')
        dist = d3.Distributor(coords, dtype=np.float64)
        basis = _basis(case, coords, dealias)
        rng = np.random.default_rng(5)
        T = dist.Field(bases=basis)
        u = dist.VectorField(coords if case != 'sphere' else coords.S2coordsys, bases=basis)
        for f in (T, u):
            f['c'] = rng.standard_normal(f['c'].shape)
            f['g']; f['c']
        out = {}
        for name, expr in EXPRESSIONS.items():
            try:
                X = expr(T, u).evaluate()
            except Exception:
                continue
            X.change_scales(dealias)
            out[name] = (X['g'].copy(), X['c'].copy())
        return out
    finally:
        fastpaths.ENABLED = old


@pytest.mark.parametrize('case', ['shell', 'ball', 'sphere'])
@pytest.mark.parametrize('dealias', [1, 3/2])
def test_fastpaths_match_original(case, dealias):
    stock = _evaluate(case, dealias, False)
    fast = _evaluate(case, dealias, True)
    stock_after = _evaluate(case, dealias, False)
    assert stock.keys() == fast.keys()
    for name in stock:
        for a, b, c in zip(stock[name], fast[name], stock_after[name]):
            scale = max(np.max(np.abs(a)), 1e-300)
            assert np.max(np.abs(a - b)) / scale < 1e-12, name
            # the fast paths must not change any shared state used by the original code
            assert np.max(np.abs(a - c)) / scale < 1e-14, name


def _ivp(enabled, nsteps=5):
    old = fastpaths.ENABLED
    fastpaths.ENABLED = enabled
    try:
        coords = d3.SphericalCoordinates('phi', 'theta', 'r')
        dist = d3.Distributor(coords, dtype=np.float64)
        shell = d3.ShellBasis(coords, shape=(16, 8, 10), radii=(0.5, 1.5), dealias=3/2, dtype=np.float64)
        s2 = shell.S2_basis()
        u = dist.VectorField(coords, bases=shell); p = dist.Field(bases=shell); T = dist.Field(bases=shell)
        tu1 = dist.VectorField(coords, bases=s2); tu2 = dist.VectorField(coords, bases=s2)
        tT1 = dist.Field(bases=s2); tT2 = dist.Field(bases=s2); tp = dist.Field()
        lift = lambda A, n: d3.Lift(A, shell.derivative_basis(1), n)
        phi, theta, r = dist.local_grids(shell)
        rvec = dist.VectorField(coords, bases=shell.radial_basis); rvec['g'][2] = r
        ez = dist.VectorField(coords, bases=shell); ez['g'][1] = -np.sin(theta); ez['g'][2] = np.cos(theta)
        gu = d3.grad(u) + rvec*lift(tu1, -1); gT = d3.grad(T) + rvec*lift(tT1, -1)
        problem = d3.IVP([p, u, T, tu1, tu2, tT1, tT2, tp], namespace=locals())
        problem.add_equation("trace(gu) + tp = 0")
        problem.add_equation("dt(u) - div(gu) + grad(p) - 100*rvec*T + lift(tu2, -1) = - u@grad(u) - 20*cross(ez, u)")
        problem.add_equation("dt(T) - div(gT) + lift(tT2, -1) = - u@gT")
        problem.add_equation("u(r=0.5) = 0"); problem.add_equation("u(r=1.5) = 0")
        problem.add_equation("T(r=0.5) = 1"); problem.add_equation("T(r=1.5) = 0")
        problem.add_equation("integ(p) = 0")
        solver = problem.build_solver(d3.CNAB2)
        rng = np.random.default_rng(3)
        T['g'] = rng.standard_normal(T['g'].shape) * 1e-2 + (1.5 - r)
        u['g'] = rng.standard_normal(u['g'].shape) * 1e-2
        for i in range(nsteps):
            solver.step(1e-3)
        out = []
        for f in (u, T):
            f.change_scales(1)
            out.append(f['c'].copy())
        return out
    finally:
        fastpaths.ENABLED = old


def test_fastpaths_ivp_steps():
    stock = _ivp(False)
    fast = _ivp(True)
    for a, b in zip(stock, fast):
        assert np.max(np.abs(a - b)) / np.max(np.abs(a)) < 1e-11
