import functools
import uuid

from libc.stdint cimport uint64_t, uint8_t, int8_t
from libc.string cimport memcpy


# A more efficient UUID type implementation
# (6-7x faster than the uuid.UUID).


cdef const char *_hexmap = b"0123456789abcdef"

cdef char _hextable[256]
_hextable[:] = [
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1, 0,1,2,3,4,5,6,7,8,9,-1,-1,-1,-1,-1,-1,-1,10,11,12,13,14,15,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,10,11,12,13,14,15,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1
]


cdef inline char i64_to_hex(uint64_t num, char *s):
    cdef:
        char i = 15
        char ch

    while i >= 0:
        s[<uint8_t>i] = _hexmap[num & 0x0F]
        num >>= 4
        i -= 1

    return 0


cdef pg_uuid_from_buf(const char *buf):
    cdef:
        PgBaseUUID u = UUID.__new__(UUID)
    memcpy(u._data, buf, 16)
    u._int = None
    return u


cdef pg_uuid_bytes_from_str(str u, char *out):
    cdef:
        char *orig_buf
        Py_ssize_t size
        unsigned char ch
        uint8_t acc, part, acc_set
        uint8_t i, j

    orig_buf = <char*>cpythonx.PyUnicode_AsUTF8AndSize(u, &size)
    if size > 36 or size < 32:
        raise ValueError(
            f'invalid UUID {u!r}: '
            f'length must be between 32..36 characters, got {size}')

    acc_set = 0
    j = 0
    for i in range(0, size):
        ch = <unsigned char>orig_buf[i]
        if ch == <unsigned char>b'-':
            continue

        part = <uint8_t><int8_t>_hextable[ch]
        if part == <uint8_t>-1:
            if ch >= 0x20 and ch <= 0x7e:
                raise ValueError(
                    f'invalid UUID {u!r}: unexpected character {chr(ch)!r}')
            else:
                raise ValueError('invalid UUID {u!r}: unexpected character')

        if acc_set:
            acc |= part
            out[j] = <char>acc
            acc_set = 0
            j += 1
        else:
            acc = <uint8_t>(part << 4)
            acc_set = 1

        if j > 16 or (j == 16 and acc_set):
            raise ValueError(
                f'invalid UUID {u!r}: decodes to more than 16 bytes')

    if j != 16:
            raise ValueError(
                f'invalid UUID {u!r}: decodes to less than 16 bytes')


cdef class PgBaseUUID:

    cdef:
        char _data[16]
        object _int

    def __init__(self, inp):
        cdef:
            char *buf
            Py_ssize_t size

        if cpython.PyBytes_Check(inp):
            cpython.PyBytes_AsStringAndSize(inp, &buf, &size)
            if size != 16:
                raise ValueError(f'16 bytes were expected, got {size}')
            memcpy(self._data, buf, 16)

        elif cpython.PyUnicode_Check(inp):
            pg_uuid_bytes_from_str(inp, self._data)
        else:
            raise TypeError(f'a bytes or str object expected, got {inp!r}')

        self._int = None

    property bytes:
        def __get__(self):
            return cpython.PyBytes_FromStringAndSize(self._data, 16)

    property int:
        def __get__(self):
            if self._int is None:
                # The cache is important because `self.int` can be
                # used multiple times by __hash__ etc.
                self._int = int.from_bytes(self.bytes, 'big')
            return self._int

    def __str__(self):
        cdef:
            uint64_t u
            char out[36]

        u = <uint64_t>hton.unpack_int64(self._data)
        i64_to_hex(u, out)
        u = <uint64_t>hton.unpack_int64(self._data + 8)
        i64_to_hex(u, out + 20)

        memcpy(out + 14, out + 12, 4)
        memcpy(out + 9, out + 8, 4)
        memcpy(out + 19, out + 20, 4)
        out[8] = b'-'
        out[13] = b'-'
        out[18] = b'-'
        out[23] = b'-'

        return cpythonx.PyUnicode_FromKindAndData(
            cpythonx.PyUnicode_1BYTE_KIND, <void*>out, 36)

    def __repr__(self):
        return f"UUID('{self}')"

    def __reduce__(self):
        return (type(self), (self.bytes,))


class UUID(PgBaseUUID, uuid.UUID):
    __slots__ = ()


cdef pg_UUID = UUID
