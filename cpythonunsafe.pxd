from cpython cimport PyObject

cdef extern from "Python.h":
    PyObject *PyBytes_FromStringAndSize(const char *, ssize_t)
    PyObject *PyUnicode_DecodeUTF8(const char *, Py_ssize_t, const char *)
    void Py_INCREF(PyObject *o)
    PyObject *PyTuple_GET_ITEM(PyObject *, Py_ssize_t) nogil
    PyObject *PyList_GET_ITEM(PyObject *, Py_ssize_t) nogil
    void *PyLong_AsVoidPtr(PyObject *pylong)
    PyObject *PyLong_FromLong(long v)
    PyObject *Py_None
    PyObject *Py_True
    PyObject *Py_False
