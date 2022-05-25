from cpython.ref cimport PyObject
from cpython.mem cimport PyMem_Malloc, PyMem_Free
import cython
from libc.stdint cimport intptr_t, int16_t, int32_t, int64_t
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset
from math import nan

import numpy as np
from numpy cimport PyArray_DATA


cdef extern from "utf8_to_ucs4.h":
    inline int utf8_to_ucs4(const char *, int32_t *, int) nogil


@cython.no_gc
@cython.final
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.freelist(_BUFFER_FREELIST_SIZE)
cdef class ArrayWriter:
    def __cinit__(self, np_dtype dtype not None):
        cdef:
            np_dtype child_dtype
            int offset, pos = 0, length = len(dtype), unit
            libdivide_s64_ex_t adjust_value
            char adjust_kind

        if not dtype.fields:
            raise ValueError("dtype must be a struct")
        self.dtype = dtype
        self.null_indexes = []
        self._dtype_length = length
        self._chunks = []
        self._recharge()
        self._dtype_kind = <char *>malloc(length)
        self._dtype_size = <int32_t *>malloc(length * 4)
        self._dtype_offset = <int32_t *>malloc((length + 1) * 4)
        self._time_adjust_value = <libdivide_s64_ex_t *>malloc(length * sizeof(libdivide_s64_ex_t))
        for i, name in enumerate(dtype.names):
            child_dtype, offset = dtype.fields[name]
            self._dtype_kind[i] = child_dtype.kind
            self._dtype_size[i] = child_dtype.itemsize
            self._dtype_offset[i] = offset - pos
            pos = offset + child_dtype.itemsize
            if child_dtype.kind == b"M" or child_dtype.kind == b"m":
                unit = (<PyArray_DatetimeDTypeMetaData *>child_dtype.c_metadata).meta.base
                if unit == NPY_FR_us or unit == NPY_FR_GENERIC:
                    adjust_value.base.magic = 1
                    adjust_value.dt_offset = 0
                elif unit == NPY_FR_ns:
                    adjust_value.base.magic = 1000
                    adjust_value.dt_offset = 0
                elif unit == NPY_FR_s:
                    adjust_value.base = libdivide_s64_gen(1000000)
                    adjust_value.dt_offset = 1 - 1000000
                elif unit == NPY_FR_ms:
                    adjust_value.base = libdivide_s64_gen(1000)
                    adjust_value.dt_offset = 1 - 1000
                elif unit == NPY_FR_D:
                    adjust_value.base = libdivide_s64_gen(24 * 3600 * 1000000)
                    adjust_value.dt_offset = 1 - 24 * 3600 * 1000000
                elif unit == NPY_FR_m:
                    adjust_value.base = libdivide_s64_gen(60 * 1000000)
                    adjust_value.dt_offset = 1 - 60 * 1000000
                elif unit == NPY_FR_h:
                    adjust_value.base = libdivide_s64_gen(3600 * 1000000)
                    adjust_value.dt_offset = 1 - 3600 * 1000000
                elif unit == NPY_FR_W:
                    adjust_value.base = libdivide_s64_gen(7 * 24 * 3600 * 1000000)
                    adjust_value.dt_offset = 1 - 7 * 24 * 3600 * 1000000
                else:
                    raise NotImplementedError(f"dtype[{i}] = {dtype[i]} time unit is not supported")
            self._time_adjust_value[i] = adjust_value
        self._dtype_offset[length] = dtype.itemsize - pos

    def __dealloc__(self):
        free(self._dtype_kind)
        free(self._dtype_size)
        free(self._dtype_offset)
        free(self._time_adjust_value)
        for ptr in self._chunks:
            PyMem_Free(<PyObject *><intptr_t>ptr)

    cdef void _step(self):
        self._data += self._dtype_size[self._field]
        self._field += 1
        self._data += self._dtype_offset[self._field]
        if self._field == self._dtype_length:
            self._field = 0
            self._item += 1
            if self._item == _ARRAY_CHUNK_SIZE:
                self._recharge()

    cdef void _recharge(self):
        self._item = 0
        self._data = <char *> PyMem_Malloc(_ARRAY_CHUNK_SIZE * self.dtype.itemsize)
        self._chunks.append(<intptr_t> self._data)

    @cython.wraparound(True)
    cdef consolidate(self):
        arr = np.empty((len(self._chunks) - 1) * _ARRAY_CHUNK_SIZE + self._item, dtype=self.dtype)
        cdef:
            char *body = <char*> PyArray_DATA(arr)
            char kind
            np_dtype dtype = self.dtype
            int64_t chunk_size = _ARRAY_CHUNK_SIZE * dtype.itemsize

        for chunk in self._chunks[:-1]:
            memcpy(body, <void*><intptr_t>chunk, chunk_size)
            PyMem_Free(<PyObject *><intptr_t>chunk)
            body += chunk_size
        memcpy(body, <void*><intptr_t>self._chunks[-1], self._item * dtype.itemsize)
        PyMem_Free(<PyObject *><intptr_t>self._chunks[-1])
        self._chunks.clear()
        return arr

    cdef raise_dtype_error(self, str pgtype, int size):
        raise DTypeError(f"dtype[{self._field}] = {self.dtype[self._field]} does not match "
                         f"PostgreSQL {pgtype}" + (f" of size {size}" if size > 0 else ""))

    cdef int current_field_is_object(self) nogil:
        return self._dtype_kind[self._field] == b"O"

    cdef int write_null(self) except -1:
        cdef:
            int i
            char dtype
            int size

        self.null_indexes.append(self._item * self._dtype_length + self._field)
        dtype = self._dtype_kind[self._field]
        size = self._dtype_size[self._field]
        if dtype == b"O":
            (<PyObject **> self._data)[0] = cpythonunsafe.Py_None
            cpythonunsafe.Py_INCREF(cpythonunsafe.Py_None)
        elif dtype == b"M" or dtype == b"m":
            (<int64_t *>self._data)[0] = (1 << 63)  # NaT
        elif dtype == b"f":
            if size == 4:
                (<float *> self._data)[0] = <float>nan
            elif size == 8:
                (<double *> self._data)[0] = nan
        elif dtype == b"V":
            if size % 8 == 0:
                for i in range(size // 8):
                    (<double *> self._data)[i] = nan
            else:
                for i in range(size):
                    (<uint8_t *> self._data)[i] = (1 << 7)
        elif dtype == b"i" or dtype == b"u" or dtype == b"b":
            if size == 1:
                (<uint8_t *> self._data)[0] = (1 << 7)
            elif size == 2:
                (<uint16_t *> self._data)[0] = (1 << 15)
            elif size == 4:
                (<uint32_t *> self._data)[0] = (1 << 31)
            elif size == 8:
                (<uint64_t *> self._data)[0] = (1 << 63)
        elif dtype == b"S":
            memset(self._data, 0xFF, size)
        elif dtype == b"U":
            memset(self._data, 0, size)
        self._step()

    cdef int write_object(self, object obj) except -1:
        cdef:
            PyObject *ptr = <PyObject *>obj
            int ret
        ret = self.write_object_unsafe(ptr)
        cpythonunsafe.Py_INCREF(ptr)
        return ret

    cdef int write_object_unsafe(self, PyObject *obj) except -1:
        if not self.current_field_is_object():
            self.raise_dtype_error("object", 0)
        (<PyObject **> self._data)[0] = obj
        self._step()

    cdef int write_bool(self, int b) except -1:
        if self._dtype_kind[self._field] != b"b":
            self.raise_dtype_error("bool", 1)
        self._data[0] = b != 0
        self._step()

    cdef int write_bytes(self, const char *data, ssize_t len) except -1:
        cdef int full_size = self._dtype_size[self._field]
        if full_size < len or self._dtype_kind[self._field] != b"S":
            self.raise_dtype_error("bytea", len)
        memcpy(self._data, data, len)
        memset(self._data + len, 0, full_size - len)
        self._step()

    cdef int write_string(self, const char *data, ssize_t len) except -1:
        cdef:
            char kind = self._dtype_kind[self._field]
            int full_size = self._dtype_size[self._field]
            int ucs4_size
        if kind == b"U":
            if full_size < 4 * len:
                self.raise_dtype_error("text", len)
            ucs4_size = 4 * utf8_to_ucs4(data, <int32_t *> self._data, len)
            memset(self._data + ucs4_size, 0, full_size - ucs4_size)
        elif kind == b"S":
            if full_size < len:
                self.raise_dtype_error("text", len)
            memcpy(self._data, data, len)
            memset(self._data + len, 0, full_size - len)
        else:
            self.raise_dtype_error("text", len)
        self._step()

    cdef int write_int16(self, int16_t i) except -1:
        cdef char kind = self._dtype_kind[self._field]
        if (kind != b"i" and kind != b"u") or self._dtype_size[self._field] != 2:
            self.raise_dtype_error("smallint", 2)
        (<int16_t *> self._data)[0] = i
        self._step()

    cdef int write_int32(self, int32_t i) except -1:
        cdef char kind = self._dtype_kind[self._field]
        if (kind != b"i" and kind != b"u") or self._dtype_size[self._field] != 4:
            self.raise_dtype_error("int", 4)
        (<int32_t *> self._data)[0] = i
        self._step()

    cdef int write_int64(self, int64_t i) except -1:
        cdef char kind = self._dtype_kind[self._field]
        if (kind != b"i" and kind != b"u") or self._dtype_size[self._field] != 8:
            self.raise_dtype_error("bigint", 8)
        (<int64_t *> self._data)[0] = i
        self._step()

    cdef int write_float(self, float f) except -1:
        if self._dtype_kind[self._field] != b"f" or self._dtype_size[self._field] != 4:
            self.raise_dtype_error("float4", 4)
        (<float *> self._data)[0] = f
        self._step()

    cdef int write_double(self, double d) except -1:
        if self._dtype_kind[self._field] != b"f" or self._dtype_size[self._field] != 8:
            self.raise_dtype_error("float8", 8)
        (<double *> self._data)[0] = d
        self._step()

    cdef int write_datetime(self, int64_t dt) except -1:
        cdef:
            libdivide_s64_ex_t *ptr = &self._time_adjust_value[self._field]
            int64_t offset = ptr.dt_offset
        if self._dtype_kind[self._field] != b"M":
            self.raise_dtype_error("timestamp", 8)
        if offset != 0:
            if dt < 0:
                dt += offset
            dt = libdivide_s64_do(dt, <libdivide_s64_t *>ptr)
        else:
            dt *= ptr.base.magic
        (<int64_t *> self._data)[0] = dt
        self._step()

    cdef int write_timedelta(self, int64_t td) except -1:
        cdef libdivide_s64_ex_t *ptr = &self._time_adjust_value[self._field]
        if self._dtype_kind[self._field] != b"m":
            self.raise_dtype_error("time", 8)
        if ptr.dt_offset != 0:
            td = libdivide_s64_do(td, <libdivide_s64_t *>ptr)
        else:
            td *= ptr.base.magic
        (<int64_t *> self._data)[0] = td
        self._step()

    cdef int write_4d(self, double high_x, double high_y, double low_x, double low_y) except -1:
        if self._dtype_kind[self._field] != b"V" or self._dtype_size[self._field] != 8 * 4:
            self.raise_dtype_error("4 of float8", 32)
        (<double *> self._data)[0] = high_x
        (<double *> self._data)[1] = high_y
        (<double *> self._data)[2] = low_x
        (<double *> self._data)[3] = low_y
        self._step()

    cdef int write_3d(self, double a, double b, double c) except -1:
        if self._dtype_kind[self._field] != b"V" or self._dtype_size[self._field] != 8 * 3:
            self.raise_dtype_error("3 of float8", 24)
        (<double *> self._data)[0] = a
        (<double *> self._data)[1] = b
        (<double *> self._data)[2] = c
        self._step()

    cdef int write_2d(self, double x, double y) except -1:
        if self._dtype_kind[self._field] != b"V" or self._dtype_size[self._field] != 8 * 2:
            self.raise_dtype_error("2 of float8", 16)
        (<double *> self._data)[0] = x
        (<double *> self._data)[1] = y
        self._step()

    cdef int write_tid(self, uint32_t block, uint16_t offset) except -1:
        if self._dtype_kind[self._field] != b"V" or self._dtype_size[self._field] != (4 + 2):
            self.raise_dtype_error("tid", 6)
        (<uint32_t *> self._data)[0] = block
        (<uint16_t *> self._data)[2] = offset
        self._step()
