# SPDX-License-Identifier: LGPL-3.0-or-later
"""ctypes declarations of the canonical geometry view used by C API checks."""

from __future__ import (
    annotations,
)

import ctypes
from typing import (
    ClassVar,
)


class IndexedGeometry(ctypes.Structure):
    _fields_ = [
        ("packed", ctypes.c_void_p),
        ("center_atom", ctypes.c_void_p),
        ("vector_scale", ctypes.c_void_p),
        ("atom_count", ctypes.c_int64),
    ]


class GeometryData(ctypes.Union):
    _fields_: ClassVar = [
        ("edge_vec", ctypes.c_void_p),
        ("indexed", IndexedGeometry),
    ]


class Geometry(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_int),
        ("index_width", ctypes.c_int),
        ("edge_index", ctypes.c_void_p),
        ("data", GeometryData),
    ]
