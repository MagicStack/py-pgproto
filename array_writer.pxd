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

    cdef void raise_dtype_error(self)
    cdef int current_field_is_object(self)
    cdef void write_null(self)
    cdef void write_object(self, object obj)
    cdef void write_object_unsafe(self, PyObject *obj)
    cdef void write_bool(self, int b)
    cdef void write_bytes(self, const char *data, ssize_t len)
    cdef void write_string(self, const char *data, ssize_t len)
    cdef void write_int16(self, int16_t i)
    cdef void write_int32(self, int32_t i)
    cdef void write_int64(self, int64_t i)
    cdef void write_float(self, float f)
    cdef void write_double(self, double d)
    cdef void write_datetime(self, int64_t dt)
    cdef void write_timedelta(self, int64_t td)
    cdef void write_4d(self, double high_x, double high_y, double low_x, double low_y)
    cdef void write_3d(self, double a, double b, double c)
    cdef void write_2d(self, double x, double y)
    cdef void write_tid(self, uint32_t block, uint16_t offset)

    cdef object consolidate(self)
    cdef void _step(self)
    cdef void _recharge(self)
