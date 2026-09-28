"""Portable loader for the unchanged C continuation recurrence.

Only library discovery differs from the original macOS-specific loader. Set
GSC_BACKEND=python for the slower reference implementation.
"""
import ctypes
import os
from pathlib import Path
import sys
import numpy as np
from fast_interval_planner import first_action

_ROOT=Path(__file__).resolve().parents[1]
_BACKEND=os.environ.get('GSC_BACKEND','native')
if _BACKEND not in ('native','python'):
    raise ValueError('GSC_BACKEND must be native or python')
if _BACKEND=='native':
    library=_ROOT/'.build'/('libcontinuation.dylib' if sys.platform=='darwin' else 'libcontinuation.so')
    if not library.exists():
        raise RuntimeError('Native planner is not built. Run: python reproduce.py build; or set GSC_BACKEND=python.')
    _LIB=ctypes.CDLL(str(library));_FN=_LIB.continuation
    _PTR=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    _FN.argtypes=[_PTR,ctypes.c_int,_PTR,ctypes.c_int,*([ctypes.c_double]*6),_PTR]
    _FN.restype=ctypes.c_int

def continuation(xs,b,states,terminal_value):
    if _BACKEND=='python':
        from fast_interval_planner import continuation as reference
        return reference(xs,b,states,terminal_value)
    levels=np.linspace(0.,b.capacity,states);value=np.empty(states,np.float64)
    values=np.asarray([(x.load,x.pv,x.price,x.dt) for x in xs],np.float64).reshape(-1,4)
    result=_FN(values,len(xs),levels,states,b.capacity/(states-1),b.charge_kw,b.discharge_kw,b.eta_charge,b.eta_discharge,terminal_value,value)
    if result:raise RuntimeError('native continuation error '+str(result))
    return levels,value

def nominal_action(x,s,b,future,states,terminal_value):
    levels,value=continuation(future,b,states,terminal_value)
    return first_action(x,s,b,levels,value)
