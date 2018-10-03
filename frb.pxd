# Copyright (C) 2016-present the asyncpg authors and contributors
# <see AUTHORS file>
#
# This module is part of asyncpg and is released under
# the Apache 2.0 License: http://www.apache.org/licenses/LICENSE-2.0


from . import exceptions


cdef:
    struct Buffer:
        const char* buf
        ssize_t len

    inline ssize_t get_len(Buffer *frb):
        return frb.len

    inline void set_len(Buffer *frb, ssize_t new_len):
        frb.len = new_len

    inline void init(Buffer *frb, const char *buf, ssize_t len):
        frb.buf = buf
        frb.len = len

    inline const char* read(Buffer *frb, ssize_t n) except NULL:
        cdef const char *result

        check(frb, n)

        result = frb.buf
        frb.buf += n
        frb.len -= n

        return result

    inline const char* read_all(Buffer *frb):
        cdef const char *result
        result = frb.buf
        frb.buf += frb.len
        frb.len = 0
        return result

    inline Buffer *slice_from(Buffer *frb, Buffer* source, ssize_t len):
        frb.buf = read(source, len)
        frb.len = len
        return frb

    inline object check(Buffer *frb, ssize_t n):
        if n > frb.len:
            raise exceptions.BufferError(
                'insufficient data in buffer: requested {}, remaining {}'.
                    format(n, frb.len))
