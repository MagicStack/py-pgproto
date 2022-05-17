from cpython cimport PyObject

cdef extern from "Python.h":
    PyObject *PyByteArray_FromStringAndSize(const char *, ssize_t)
    PyObject *PyUnicode_DecodeUTF8(const char *, Py_ssize_t, const char *)
    void Py_INCREF(PyObject *o)
    PyObject *PyTuple_GET_ITEM(PyObject *, Py_ssize_t)
    PyObject *Py_None
