import functools
import uuid

from libc.stdint cimport uint64_t, uint8_t, int8_t
from libc.string cimport memcpy, strncmp


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


cdef class __UUIDReplaceMe:
    pass


cdef pg_uuid_from_buf(const char *buf):
    cdef:
        UUID u = UUID.__new__(UUID)
    memcpy(u._data, buf, 16)
    return u


@cython.final
@cython.no_gc_clear
cdef class UUID(__UUIDReplaceMe):

    cdef:
        char _data[16]
        object _int
        object _hash
        object __weakref__

    def __cinit__(self):
        self._int = None
        self._hash = None

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

    @property
    def bytes(self):
        return cpython.PyBytes_FromStringAndSize(self._data, 16)

    @property
    def int(self):
        if self._int is None:
            # The cache is important because `self.int` can be
            # used multiple times by __hash__ etc.
            self._int = int.from_bytes(self.bytes, 'big')
        return self._int

    @property
    def is_safe(self):
        return uuid.SafeUUID.unknown

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

    def __eq__(self, other):
        if type(other) is UUID:
            return strncmp(self._data, (<UUID>other)._data, 16) == 0
        if isinstance(other, uuid.UUID):
            return self.int == other.int
        return NotImplemented

    def __ne__(self, other):
        if type(other) is UUID:
            return strncmp(self._data, (<UUID>other)._data, 16) != 0
        if isinstance(other, uuid.UUID):
            return self.int != other.int
        return NotImplemented

    def __lt__(self, other):
        if type(other) is UUID:
            return strncmp(self._data, (<UUID>other)._data, 16) < 0
        if isinstance(other, uuid.UUID):
            return self.int < other.int
        return NotImplemented

    def __gt__(self, other):
        if type(other) is UUID:
            return strncmp(self._data, (<UUID>other)._data, 16) > 0
        if isinstance(other, uuid.UUID):
            return self.int > other.int
        return NotImplemented

    def __le__(self, other):
        if type(other) is UUID:
            return strncmp(self._data, (<UUID>other)._data, 16) <= 0
        if isinstance(other, uuid.UUID):
            return self.int <= other.int
        return NotImplemented

    def __ge__(self, other):
        if type(other) is UUID:
            return strncmp(self._data, (<UUID>other)._data, 16) >= 0
        if isinstance(other, uuid.UUID):
            return self.int >= other.int
        return NotImplemented

    def __hash__(self):
        # In EdgeDB every schema object has a uuid and there are
        # huge hash-maps of them. We want UUID.__hash__ to be
        # as fast as possible.
        if self._hash is not None:
            return self._hash

        self._hash = hash(self.int)
        return self._hash

    def __int__(self):
        return self.int

    @property
    def bytes_le(self):
        bytes = self.bytes
        return (bytes[4-1::-1] + bytes[6-1:4-1:-1] + bytes[8-1:6-1:-1] +
                bytes[8:])

    @property
    def fields(self):
        return (self.time_low, self.time_mid, self.time_hi_version,
                self.clock_seq_hi_variant, self.clock_seq_low, self.node)

    @property
    def time_low(self):
        return self.int >> 96

    @property
    def time_mid(self):
        return (self.int >> 80) & 0xffff

    @property
    def time_hi_version(self):
        return (self.int >> 64) & 0xffff

    @property
    def clock_seq_hi_variant(self):
        return (self.int >> 56) & 0xff

    @property
    def clock_seq_low(self):
        return (self.int >> 48) & 0xff

    @property
    def time(self):
        return (((self.time_hi_version & 0x0fff) << 48) |
                (self.time_mid << 32) | self.time_low)

    @property
    def clock_seq(self):
        return (((self.clock_seq_hi_variant & 0x3f) << 8) |
                self.clock_seq_low)

    @property
    def node(self):
        return self.int & 0xffffffffffff

    @property
    def hex(self):
        cdef:
            uint64_t u
            char out[32]

        u = <uint64_t>hton.unpack_int64(self._data)
        i64_to_hex(u, out)
        u = <uint64_t>hton.unpack_int64(self._data + 8)
        i64_to_hex(u, out + 16)

        return cpythonx.PyUnicode_FromKindAndData(
            cpythonx.PyUnicode_1BYTE_KIND, <void*>out, 32)

    @property
    def urn(self):
        return 'urn:uuid:' + str(self)

    @property
    def variant(self):
        if not self.int & (0x8000 << 48):
            return uuid.RESERVED_NCS
        elif not self.int & (0x4000 << 48):
            return uuid.RFC_4122
        elif not self.int & (0x2000 << 48):
            return uuid.RESERVED_MICROSOFT
        else:
            return uuid.RESERVED_FUTURE

    @property
    def version(self):
        # The version bits are only meaningful for RFC 4122 UUIDs.
        if self.variant == uuid.RFC_4122:
            return int((self.int >> 76) & 0xf)


# <hack>
# In order for `isinstance(pgproto.UUID, uuid.UUID)` to work,
# patch __bases__ and __mro__ by injecting `uuid.UUID`.
#
# We apply brute-force here because the following pattern stopped
# working with Python 3.8:
#
#   cdef class OurUUID:
#       ...
#
#   class UUID(OurUUID, uuid.UUID):
#       ...
#
# With Python 3.8 it now produces
#
#   "TypeError: multiple bases have instance lay-out conflict"
#
# error.  Maybe it's possible to fix this some other way, but
# the best solution possible would be to just contribute our
# faster UUID to the standard library and not have this problem
# at all.  For now this hack is pretty safe and should be
# compatible with future Pythons for long enough.
#
assert UUID.__bases__[0] is __UUIDReplaceMe
assert UUID.__mro__[1] is __UUIDReplaceMe
cpython.Py_INCREF(uuid.UUID)
cpython.PyTuple_SET_ITEM(UUID.__bases__, 0, uuid.UUID)
cpython.Py_INCREF(uuid.UUID)
cpython.PyTuple_SET_ITEM(UUID.__mro__, 1, uuid.UUID)
# </hack>


cdef pg_UUID = UUID
