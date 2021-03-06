// Author: Sudeep Pillai (spillai@csail.mit.edu)
// Note: Stripped from Opencv (opencv/modules/python/src2/cv2.cpp)

# ifndef __COVERSION_OPENCV_H__
# define __COVERSION_OPENCV_H__

#include <Python.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <numpy/ndarrayobject.h>

class PyAllowThreads;

class PyEnsureGIL;

static size_t REFCOUNT_OFFSET = (size_t)&(((PyObject*)0)->ob_refcnt) +
    (0x12345678 != *(const size_t*)"\x78\x56\x34\x12\0\0\0\0\0")*sizeof(int);

static inline PyObject* pyObjectFromRefcount(const int* refcount)
{
    return (PyObject*)((size_t)refcount - REFCOUNT_OFFSET);
}

static inline int* refcountFromPyObject(const PyObject* obj)
{
    return (int*)((size_t)obj + REFCOUNT_OFFSET);
}


class NumpyAllocator;

enum { ARG_NONE = 0, ARG_MAT = 1, ARG_SCALAR = 2 };

class NDArrayConverter
{
private:
    int init();
public:
    NDArrayConverter();
    cv::Mat to_mat(const PyObject* o);
    PyObject* to_array(const cv::Mat& mat);
};

# endif
