"""
Common-subexpression elimination for evaluator handler tasks.

Operators built separately from the same arguments (e.g. grad(u) written in several equations, or once in a Python
expression and again in an equation string) are separate Future objects, and each is evaluated separately, including
its transforms and transposes. Every Future records the arguments of its constructor (see Future.__init_subclass__).
Before evaluation, the task trees of all handlers are walked bottom-up, and a Future whose class and constructor
arguments (with operand Futures replaced by their merged instances) match an earlier one is replaced in its parents'
arguments by that earlier one. With store_last, a shared Future is then evaluated once per evaluation, so the results
are bitwise identical to evaluating the copies separately.

Fields, bases, coordinates, arrays and functions are compared by identity, numbers and strings by value. Futures with
an explicit out, tangent or cotangent, Futures with store_last disabled, and GeneralFunction (arbitrary, possibly
impure user code) are never merged. Top-level task operators are kept; only their subtrees are merged.

Switch off with [optimization] COMMON_SUBEXPRESSIONS = False in dedalus.cfg.
"""

import numbers
import numpy as np

from ..tools.config import config

import logging
logger = logging.getLogger(__name__.split('.')[-1])

ENABLED = config['optimization'].getboolean('COMMON_SUBEXPRESSIONS', fallback=False)


def _key(x, canon):
    """Hashable key for a constructor argument."""
    from .future import Future
    if isinstance(x, Future):
        return ('F', id(canon(x)))
    if x is None or isinstance(x, (bool, str)):
        return ('v', type(x), x)
    if isinstance(x, (numbers.Number, np.generic)):
        return ('n', type(x), x)
    if isinstance(x, (tuple, list)):
        return ('t', type(x), tuple(_key(y, canon) for y in x))
    if isinstance(x, dict):
        return ('d', tuple(sorted((k, _key(v, canon)) for k, v in x.items())))
    return ('o', id(x))


def _substitute(x, canon):
    """Constructor argument x with every Future replaced by its merged instance."""
    from .future import Future
    if isinstance(x, Future):
        return canon(x)
    if isinstance(x, (tuple, list)):
        return type(x)(_substitute(y, canon) for y in x)
    if isinstance(x, dict):
        return {k: _substitute(v, canon) for k, v in x.items()}
    return x


def eliminate(handlers):
    """Merge identical subexpressions across the tasks of the given handlers. Returns the number of merged Futures."""
    from .future import Future
    from .operators import GeneralFunction
    table = {}       # key -> canonical Future
    memo = {}        # id(Future) -> canonical Future
    keep = []        # keep every visited Future alive while its id is in use
    merged = 0

    def canon(f):
        nonlocal merged
        i = id(f)
        if i in memo:
            return memo[i]
        memo[i] = f
        keep.append(f)
        # Merge the operands first
        new_args = [canon(a) if isinstance(a, Future) else a for a in f.args]
        if any(n is not o for n, o in zip(new_args, f.args)):
            replace = {id(o): n for n, o in zip(new_args, f.args) if n is not o}
            f.args = new_args
            f.original_args = tuple(replace.get(id(a), a) for a in f.original_args)
            operand = f.__dict__.get('operand')
            if operand is not None and id(operand) in replace:
                f.operand = replace[id(operand)]
        construction = f.__dict__.get('_construction')
        if construction is None:
            return f
        # Constructor arguments in terms of merged Futures (also drops references to replaced copies)
        cls, args, kw = construction
        args, kw = _substitute(tuple(args), canon), _substitute(dict(kw), canon)
        f._construction = (cls, args, kw)
        if not f.store_last or isinstance(f, GeneralFunction):
            return f
        if any(kw.get(k) is not None for k in ('out', 'tangent', 'cotangent')):
            return f
        try:
            key = (cls, _key(args, canon), _key(kw, canon))
            hash(key)
        except TypeError:
            return f
        c = table.get(key)
        if c is None:
            table[key] = f
            return f
        memo[i] = c
        merged += 1
        return c

    for handler in handlers:
        for task in handler.tasks:
            if isinstance(task['operator'], Future):
                canon(task['operator'])
    return merged
