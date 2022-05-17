from cpython.ref cimport PyObject
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.stdint cimport intptr_t
from libc.string cimport memcpy, memset
from math import nan

import numpy as np
from numpy cimport PyArray_DATA

@cython.no_gc
@cython.final
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.freelist(_BUFFER_FREELIST_SIZE)
cdef class ArrayWriter:
    def __cinit__(self, object dtype):
        if not isinstance(dtype, np.dtype) or dtype.kind != "V":
            raise ValueError("dtype must be a struct (V)")
        self.dtype = dtype
        self.null_indexes = []
        self._chunks = []
        self._recharge()
        self._dtype_kind = np.empty(len(dtype), dtype=np.byte)
        self._dtype_size = np.empty(len(dtype), dtype=np.int32)
        self._dtype_offset = np.empty(len(dtype) + 1, dtype=np.int32)
        cdef int pos = 0
        for i, name in enumerate(dtype.names):
            child_dtype, offset = dtype.fields[name]
            self._dtype_kind[i] = cpythonx.PyUnicode_AsUTF8AndSize(child_dtype.kind, NULL)[0]
            self._dtype_size[i] = child_dtype.itemsize
            self._dtype_offset[i] = offset - pos
            pos = offset + child_dtype.itemsize
        self._dtype_offset[-1] = dtype.itemsize - pos

    def __dealloc__(self):
        for ptr in self._chunks:
            PyMem_Free(<PyObject *><intptr_t>ptr)

    cdef void _step(self):
        self._data += self._dtype_size[self._field]
        self._field += 1
        self._data += self._dtype_offset[self._field]
        if self._field == len(self._dtype_kind):
            self._field = 0
            self._item += 1
            if self._item == _ARRAY_CHUNK_SIZE:
                self._recharge()

    cdef void _recharge(self):
        self._item = 0
        self._data = <char *> PyMem_Malloc(_ARRAY_CHUNK_SIZE * self.dtype.itemsize)
        self._chunks.append(<intptr_t> self._data)

    cdef object consolidate(self):
        arr = np.empty((len(self._chunks) - 1) * _ARRAY_CHUNK_SIZE + self._item, dtype=self.dtype)
        cdef:
            char *body = <char*> PyArray_DATA(arr)
            int64_t chunk_size = _ARRAY_CHUNK_SIZE * self.dtype.itemsize
            char kind

        for chunk in self._chunks[:-1]:
            memcpy(body, <void*><intptr_t>chunk, chunk_size)
            PyMem_Free(<PyObject *><intptr_t>chunk)
            body += chunk_size
        memcpy(body, <void*><intptr_t>self._chunks[-1], self._item * self.dtype.itemsize)
        PyMem_Free(<PyObject *><intptr_t>self._chunks[-1])
        self._chunks.clear()

        # adjust datetime64 and timedelta64 units
        dtype_datetime64_us = np.dtype("datetime64[us]")
        dtype_timedelta64_us = np.dtype("timedelta64[us]")
        for i in range(len(self.dtype)):
            kind = self._dtype_kind[i]
            if kind == b"M":
                if self.dtype[i] != dtype_datetime64_us:
                    name = self.dtype.names[i]
                    arr[name] = arr[name].view(dtype_datetime64_us).astype(self.dtype[i])
            elif kind == b"m":
                if self.dtype[i] != dtype_timedelta64_us:
                    name = self.dtype.names[i]
                    arr[name] = arr[name].view(dtype_timedelta64_us).astype(self.dtype[i])
        return arr

    cdef void raise_dtype_error(self):
        raise DTypeError(self._field)

    cdef int current_field_is_object(self):
        return self._dtype_kind[self._field] == b"O"

    cdef void write_null(self):
        cdef:
            int i
            char dtype
            int size

        self.null_indexes.append(self._item * len(self._dtype_kind) + self._field)
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
        elif dtype == b"S" or dtype == b"U":
            memset(self._data, 0xFF, size)
        self._step()

    cdef void write_object(self, object obj):
        cdef PyObject *ptr = <PyObject *>obj;
        self.write_object_unsafe(ptr)
        cpythonunsafe.Py_INCREF(ptr)

    cdef void write_object_unsafe(self, PyObject *obj):
        if not self.current_field_is_object():
            self.raise_dtype_error()
        (<PyObject **> self._data)[0] = obj
        self._step()

    cdef void write_bool(self, int b):
        if self._dtype_kind[self._field] != b"b":
            self.raise_dtype_error()
        self._data[0] = b != 0
        self._step()

    cdef void write_bytes(self, const char *data, ssize_t len):
        if self._dtype_size[self._field] < len or self._dtype_kind[self._field] != b"S":
            self.raise_dtype_error()
        memcpy(self._data, data, len)
        self._step()

    cdef void write_string(self, const char *data, ssize_t len):
        cdef char kind = self._dtype_kind[self._field]
        if kind != b"S" and kind != b"U":
            self.raise_dtype_error()
        if kind == b"U":
            if self._dtype_size[self._field] < 4 * len:
                self.raise_dtype_error()
        else:
            if self._dtype_size[self._field] < len:
                self.raise_dtype_error()
        memcpy(self._data, data, len)
        self._step()

    cdef void write_int16(self, int16_t i):
        cdef char kind = self._dtype_kind[self._field]
        if (kind != b"i" and kind != b"u") or self._dtype_size[self._field] != 2:
            self.raise_dtype_error()
        (<int16_t *> self._data)[0] = i
        self._step()

    cdef void write_int32(self, int32_t i):
        cdef char kind = self._dtype_kind[self._field]
        if (kind != b"i" and kind != b"u") or self._dtype_size[self._field] != 4:
            self.raise_dtype_error()
        (<int32_t *> self._data)[0] = i
        self._step()

    cdef void write_int64(self, int64_t i):
        cdef char kind = self._dtype_kind[self._field]
        if (kind != b"i" and kind != b"u") or self._dtype_size[self._field] != 8:
            self.raise_dtype_error()
        (<int64_t *> self._data)[0] = i
        self._step()

    cdef void write_float(self, float f):
        if self._dtype_kind[self._field] != b"f" or self._dtype_size[self._field] != 4:
            self.raise_dtype_error()
        (<float *> self._data)[0] = f
        self._step()

    cdef void write_double(self, double d):
        if self._dtype_kind[self._field] != b"f" or self._dtype_size[self._field] != 8:
            self.raise_dtype_error()
        (<double *> self._data)[0] = d
        self._step()

    cdef void write_datetime(self, int64_t dt):
        if self._dtype_kind[self._field] != b"M":
            self.raise_dtype_error()
        (<int64_t *> self._data)[0] = dt
        self._step()

    cdef void write_timedelta(self, int64_t td):
        if self._dtype_kind[self._field] != b"m":
            self.raise_dtype_error()
        (<int64_t *> self._data)[0] = td
        self._step()

    cdef void write_4d(self, double high_x, double high_y, double low_x, double low_y):
        if self._dtype_kind[self._field] != b"V" or self._dtype_size[self._field] != 8 * 4:
            self.raise_dtype_error()
        (<double *> self._data)[0] = high_x
        (<double *> self._data)[1] = high_y
        (<double *> self._data)[2] = low_x
        (<double *> self._data)[3] = low_y
        self._step()

    cdef void write_3d(self, double a, double b, double c):
        if self._dtype_kind[self._field] != b"V" or self._dtype_size[self._field] != 8 * 3:
            self.raise_dtype_error()
        (<double *> self._data)[0] = a
        (<double *> self._data)[1] = b
        (<double *> self._data)[2] = c
        self._step()

    cdef void write_2d(self, double x, double y):
        if self._dtype_kind[self._field] != b"V" or self._dtype_size[self._field] != 8 * 2:
            self.raise_dtype_error()
        (<double *> self._data)[0] = x
        (<double *> self._data)[1] = y
        self._step()

    cdef void write_tid(self, uint32_t block, uint16_t offset):
        if self._dtype_kind[self._field] != b"V" or self._dtype_size[self._field] != (4 + 2):
            self.raise_dtype_error()
        (<uint32_t *> self._data)[0] = block
        (<uint16_t *> self._data)[2] = offset
        self._step()
