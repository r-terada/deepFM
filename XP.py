from __future__ import print_function, unicode_literals, division
from chainer import cuda, Variable
import numpy as np


class XP:
    __lib = None

    @staticmethod
    def set_library(gpu_device):
        if gpu_device >= 0:
            XP.__lib = cuda.cupy
            cuda.get_device(gpu_device).use()
        else:
            XP.__lib = np

    @staticmethod
    def get_library():
        return XP.__lib

    @staticmethod
    def __zeros(shape, dtype):
        return Variable(XP.__lib.zeros(shape, dtype=dtype))

    @staticmethod
    def fzeros(shape):
        return XP.__zeros(shape, XP.__lib.float32)

    @staticmethod
    def __ones(shape, dtype):
        return Variable(XP.__lib.ones(shape, dtype=dtype))

    @staticmethod
    def fones(shape):
        return XP.__ones(shape, XP.__lib.float32)

    @staticmethod
    def iones(shape):
        return XP.__ones(shape, XP.__lib.int32)

    @staticmethod
    def __array(array, dtype):
        return Variable(XP.__lib.array(array, dtype=dtype))

    @staticmethod
    def iarray(array):
        return XP.__array(array, XP.__lib.int32)

    @staticmethod
    def farray(array):
        return XP.__array(array, XP.__lib.float32)
