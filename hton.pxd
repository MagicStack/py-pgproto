# Copyright (C) 2016-present the asyncpg authors and contributors
# <see AUTHORS file>
#
# This module is part of asyncpg and is released under
# the Apache 2.0 License: http://www.apache.org/licenses/LICENSE-2.0


from libc.stdint cimport int16_t, int32_t, uint16_t, uint32_t, int64_t, uint64_t


cdef extern from "./hton.h":
    cdef void pack_int16(char *buf, int16_t x) nogil;
    cdef void pack_int32(char *buf, int32_t x) nogil;
    cdef void pack_int64(char *buf, int64_t x) nogil;
    cdef void pack_float(char *buf, float f) nogil;
    cdef void pack_double(char *buf, double f) nogil;
    cdef int16_t unpack_int16(const char *buf) nogil;
    cdef uint16_t unpack_uint16(const char *buf) nogil;
    cdef int32_t unpack_int32(const char *buf) nogil;
    cdef uint32_t unpack_uint32(const char *buf) nogil;
    cdef int64_t unpack_int64(const char *buf) nogil;
    cdef uint64_t unpack_uint64(const char *buf) nogil;
    cdef float unpack_float(const char *buf) nogil;
    cdef double unpack_double(const char *buf) nogil;
