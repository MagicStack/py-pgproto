from cpython cimport PyObject
from numpy cimport dtype as np_dtype


cdef class DTypeError(Exception):
    pass


cdef class ArrayWriter:
    cdef:
        np_dtype dtype
        list null_indexes
        list _chunks
        char[:] _dtype_kind
        int32_t[:] _dtype_size
        int32_t[:] _dtype_offset
        int64_t _item
        int16_t _field
        char *_data

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
    cdef void _step(self)
    cdef void _recharge(self)
