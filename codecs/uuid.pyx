# Copyright (C) 2016-present the asyncpg authors and contributors
# <see AUTHORS file>
#
# This module is part of asyncpg and is released under
# the Apache 2.0 License: http://www.apache.org/licenses/LICENSE-2.0


import uuid


_UUID = uuid.UUID


cdef uuid_encode(CodecContext settings, WriteBuffer wbuf, obj):
    if cpython.PyUnicode_Check(obj):
        obj = _UUID(obj)

    bytea_encode(settings, wbuf, obj.bytes)


cdef uuid_decode(CodecContext settings, FRBuffer *buf):
    return _UUID(bytes=bytea_decode(settings, buf))
