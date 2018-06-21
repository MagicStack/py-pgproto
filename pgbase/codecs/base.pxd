# Copyright (C) 2016-present the asyncpg authors and contributors
# <see AUTHORS file>
#
# This module is part of asyncpg and is released under
# the Apache 2.0 License: http://www.apache.org/licenses/LICENSE-2.0


cdef class CodecContext:

    cpdef get_text_codec(self)
    cdef is_encoding_utf8(self)


ctypedef object (*encode_func)(CodecContext settings,
                               WriteBuffer buf,
                               object obj)

ctypedef object (*decode_func)(CodecContext settings,
                               FastReadBuffer buf)
