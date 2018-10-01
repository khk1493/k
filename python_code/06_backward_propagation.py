import torch
from torch.autograd import Variable
import numpy as np

x_data = [1.0, 2.0, 3.0];  y_data = [2.0, 4.0, 6.0]
w = Variable(torch.Tensor([1.0]), requires_grad=True)
w1 = Variable(torch.Tensor([1.0]), requires_grad=True)
r = 0.03

def forward(x, w):
    return x * w


def loss(x, w, y):
    y_pred = forward(x, w)
    return (y_pred - y) ** 2


for epoch in range(5):
    print("epoch = ", epoch)

    for x_val, y_val in zip(x_data, y_data):
        print("\n\tx = ", x_val, "y = ", y_val)

        # forward
        L1 = forward(x_val, w); print("L1 = ", L1.data)

        # backward propagation - 2nd layer
        l2 = loss(L1, w1, y_val)
        print("\tloss2_start = ", l2.data)

        l2.backward() # dl/dw
        print("\tw1_gradient = ", w1.grad.data)

        w1.data = w1.data - r * w1.grad.data
        print("\tw1.data = ", w1.data, "\n")

        # backward propagation - 1st layer
        L11 = forward(x_val, w1)
        l1 = loss(x_val, w, L11)
        print("\tloss1_start = ", l1.data)

        l1.backward() # dl/dw
        print("\tw_gradient = ", w.grad.data)

        w.data = w.data - r * w.grad.data
        print("\tw.data = ", w.data, "\n")

        w1.grad.data.zero_()
        w.grad.data.zero_()


l2 = l2.data.numpy()
l2 = np.around(l2, decimals = 5)
print("\nloss2_end = ", l2)

print("w = ", w.data)
print("w1 = ", w1.data)

x = 4
y_hat = x*w*w1
print("y_hat = ", y_hat)