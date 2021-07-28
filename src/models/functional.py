
def optimal_padding(in_shape, kernel_size, stride=1):
    return int(((stride - 1)* in_shape - stride + kernel_size) // 2)