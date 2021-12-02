import numpy


def max_zero(array: numpy.ndarray):
    non_zero_array = [max(0, i) for i in array]
    non_zero_array = numpy.array(non_zero_array)
    return non_zero_array
