import cupy as cp
from layer import Layer_Dense


def spiral_data(points, classes):
    X = cp.zeros((points*classes, 2))
    y = cp.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = cp.linspace(0.0, 1, points)  # radius
        t = cp.linspace(class_number*4, (class_number+1)*4, points) + cp.random.randn(points)*0.2
        X[ix] = cp.c_[r*cp.sin(t*2.5), r*cp.cos(t*2.5)]
        y[ix] = class_number
    return X, y


X,y = spiral_data(3,1)

layer1 = Layer_Dense(2,100)
output_layer = Layer_Dense(100,3,"SoftMax")

print(X)

layer1.forward(X)
output_layer.forward(layer1.output)


print(output_layer.output)


