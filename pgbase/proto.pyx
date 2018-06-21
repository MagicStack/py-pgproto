# Copyright (C) 2016-present the asyncpg authors and contributors
# <see AUTHORS file>
#
# This module is part of asyncpg and is released under
# the Apache 2.0 License: http://www.apache.org/licenses/LICENSE-2.0


cimport cython

from .python cimport (
    PyMem_Malloc, PyMem_Realloc, PyMem_Calloc, PyMem_Free,
    PyMemoryView_GET_BUFFER, PyMemoryView_Check,
    PyMemoryView_FromMemory, PyMemoryView_GetContiguous,
    PyUnicode_AsUTF8AndSize, PyByteArray_AsString,
    PyByteArray_Check, PyUnicode_AsUCS4Copy,
    PyByteArray_Size, PyByteArray_Resize,
    PyByteArray_FromStringAndSize,
    PyUnicode_FromKindAndData, PyUnicode_4BYTE_KIND,
    PyUnicode_FromString, PyUnicode_FromStringAndSize
)


from libc.stdint cimport int8_t, uint8_t, int16_t, uint16_t, \
                         int32_t, uint32_t, int64_t, uint64_t, \
                         INT16_MIN, INT16_MAX, INT32_MIN, INT32_MAX, \
                         UINT32_MAX, INT64_MIN, INT64_MAX


from . cimport hton
from .debug cimport PG_DEBUG


include "./consts.pxi"
include "./buffer.pyx"

include "./codecs/base.pyx"

include "./codecs/bytea.pyx"
include "./codecs/text.pyx"

include "./codecs/datetime.pyx"
include "./codecs/float.pyx"
include "./codecs/int.pyx"
include "./codecs/json.pyx"
include "./codecs/uuid.pyx"
include "./codecs/numeric.pyx"
include "./codecs/bits.pyx"
include "./codecs/geometry.pyx"
include "./codecs/hstore.pyx"
include "./codecs/misc.pyx"
include "./codecs/network.pyx"
include "./codecs/tid.pyx"
include "./codecs/txid.pyx"
