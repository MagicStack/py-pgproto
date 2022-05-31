from cpython cimport PyObject
from libc.stdint cimport uint8_t, int64_t


cdef enum NPY_DATETIMEUNIT:
    # Force signed enum type, must be -1 for code compatibility
    NPY_FR_ERROR = -1,  # error or undetermined

    # Start of valid units
    NPY_FR_Y = 0,  # Years
    NPY_FR_M = 1,  # Months
    NPY_FR_W = 2,  # Weeks
    # Gap where 1.6 NPY_FR_B (value 3) was
    NPY_FR_D = 4,  # Days
    NPY_FR_h = 5,  # hours
    NPY_FR_m = 6,  # minutes
    NPY_FR_s = 7,  # seconds
    NPY_FR_ms = 8,  # milliseconds
    NPY_FR_us = 9,  # microseconds
    NPY_FR_ns = 10,  # nanoseconds
    NPY_FR_ps = 11,  # picoseconds
    NPY_FR_fs = 12,  # femtoseconds
    NPY_FR_as = 13,  # attoseconds
    NPY_FR_GENERIC = 14  # unbound units, can convert to anything


cdef extern from "numpy/arrayobject.h":
    ctypedef struct PyArray_DatetimeMetaData:
        NPY_DATETIMEUNIT base

    ctypedef struct PyArray_DatetimeDTypeMetaData:
        PyArray_DatetimeMetaData meta

    ctypedef class numpy.dtype[object PyArray_Descr, check_size ignore]:
        cdef:
            char kind
            int itemsize "elsize"
            dict fields
            tuple names
            void *c_metadata


ctypedef dtype np_dtype

cdef enum StorageMajor:
    kRowMajor
    kColumnMajor


cdef extern from "numpy/libdivide/libdivide.h":
    struct libdivide_s64_t:
        int64_t magic
        uint8_t more

    libdivide_s64_t libdivide_s64_gen(int64_t d) nogil
    int64_t libdivide_s64_do(int64_t numer, const libdivide_s64_t *denom) nogil


cdef class DTypeError(Exception):
    pass


cdef struct libdivide_s64_ex_t:
    libdivide_s64_t base
    int64_t dt_offset


cdef class ArrayWriter:
    cdef:
        np_dtype dtype
        StorageMajor major
        list null_indexes
        list _chunks
        int _dtype_length
        char *_dtype_kind
        uint32_t *_dtype_size
        uint32_t *_dtype_offset
        libdivide_s64_ex_t *_time_adjust_value
        int64_t _item
        int16_t _field
        char *_data
        char *_chunk

    cdef raise_dtype_error(self, str pgtype, int size)
    cdef int current_field_is_object(self) nogil
    cdef int write_null(self) except -1
    cdef int write_object(self, object obj) except -1
    cdef int write_object_unsafe(self, PyObject *obj) except -1
    cdef int write_bool(self, int b) except -1
    cdef int write_bytes(self, const char *data, ssize_t len) except -1
    cdef int write_string(self, const char *data, ssize_t len) except -1
    cdef int write_int16(self, int16_t i) except -1
    cdef int write_int32(self, int32_t i) except -1
    cdef int write_int64(self, int64_t i) except -1
    cdef int write_float(self, float f) except -1
    cdef int write_double(self, double d) except -1
    cdef int write_datetime(self, int64_t dt) except -1
    cdef int write_timedelta(self, int64_t td) except -1
    cdef int write_4d(self, double high_x, double high_y, double low_x, double low_y) except -1
    cdef int write_3d(self, double a, double b, double c) except -1
    cdef int write_2d(self, double x, double y) except -1
    cdef int write_tid(self, uint32_t block, uint16_t offset) except -1

    cdef consolidate(self)
    cdef _consolidate_row_major(self)
    cdef _consolidate_column_major(self)
    cdef void _step(self)
    cdef void _recharge(self)
