import pickle
import numpy as np
import matplotlib.pyplot as plt
from MemoryNetwork import MemoryNeuralNetwork

def evaluate_plant(y, u):
    y1, y2, y3 = y[0], y[1], y[2]
    u1, u2 = u[0], u[1]
    plant_output = (y1 * y2 * y3 * u2 * (y3 - 1) + u1)/(1 + y2 ** 2 + y3 ** 2)
    return plant_output

filename = open("trained_models/trained_mnn.obj", "rb")
mnn = pickle.load(filename)
filename.close()

timesteps = 1050
y = [0.0] * 4
u = [0.0] * 3

x_axis = []
yp_axis = []
yMNN_axis = []

for i in range(timesteps):
    if(i < 50):
        u[0] = 0.0
        y[0] = evaluate_plant(y[1:], u[1:])
        mnn.feedforward([y[1], u[0]])
        y[1:] = y[:3]
        u[1:] = u[:2]
        x_axis.append(i)
        yp_axis.append(y[0])
        yMNN_axis.append(mnn.output_nn)
    
    elif(i < 300):
        u[0] = float(np.sin(np.pi * i / 25.0))
        y[0] = evaluate_plant(y[1:], u[1:])
        mnn.feedforward([y[1], u[0]])
        y[1:] = y[:3]
        u[1:] = u[:2]
        x_axis.append(i)
        yp_axis.append(y[0])
        yMNN_axis.append(mnn.output_nn)
    elif(i < 550):
        u[0] = 1.0
        y[0] = evaluate_plant(y[1:], u[1:])
        mnn.feedforward([y[1], u[0]])
        y[1:] = y[:3]
        u[1:] = u[:2]
        x_axis.append(i)
        yp_axis.append(y[0])
        yMNN_axis.append(mnn.output_nn)
    elif(i < 800):
        u[0] = -1.0
        y[0] = evaluate_plant(y[1:], u[1:])
        mnn.feedforward([y[1], u[0]])
        y[1:] = y[:3]
        u[1:] = u[:2]
        x_axis.append(i)
        yp_axis.append(y[0])
        yMNN_axis.append(mnn.output_nn)
    else:
        u[0] = float((0.3 * np.sin(np.pi * i / 25.0)) + (0.1 * np.sin(np.pi * i / 32.0)) + (0.6 * np.sin(np.pi * i / 10.0)))
        y[0] = evaluate_plant(y[1:], u[1:])
        mnn.feedforward([y[1], u[0]])
        y[1:] = y[:3]
        u[1:] = u[:2]
        x_axis.append(i)
        yp_axis.append(y[0])
        yMNN_axis.append(mnn.output_nn)
    
plt.xlabel("Timesteps")
plt.ylabel("Output")
plt.xlim(0.0, 1050.0)
plt.title("Output of Network vs Output of plant")
plt.plot(x_axis, yp_axis, x_axis, yMNN_axis, 'r--', linewidth = 0.8)
plt.grid(ls = '-.', lw = 0.4)
plt.show()
