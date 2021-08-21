import pickle
import numpy as np
from MemoryNetwork import MemoryNeuralNetwork

def evaluate_plant(y, u):
    y1, y2, y3 = y[0], y[1], y[2]
    u1, u2 = u[0], u[1]
    plant_output = (y1 * y2 * y3 * u1 * (y3 - 1) + u1)/(1 + y2 ** 2 + y3 ** 2)
    return plant_output
    

def main():
    np.random.seed(103)
    mnn = MemoryNeuralNetwork(2, 6, 1, 0.5, 0.5)
    timesteps = 77000
    is_unstable = False
    y = [0.0] * 4             #y = [y, y1, y2, y3]
    u = [0.0] * 3             #u = [u, u1, u2]
    print("Training the Network...")
    
    for i in range(timesteps):
        if(mnn.squared_error > 1e5):
            print("Network is unstable!")
            is_unstable = True
            break
        
        if (i < 2000):
            u[0] = 0.0
            y[0] = evaluate_plant(y[1:], u[1:])
            mnn.feedforward([y[1], u[0]])
            mnn.backprop(y[0])
            y[1:] = y[:3]
            u[1:] = u[:2]
            
        elif (i < 52000):
            u[0] = 4.0 * float(np.random.rand()) - 2.0
            y[0] = evaluate_plant(y[1:], u[1:])
            mnn.feedforward([y[1], u[0]])
            mnn.backprop(y[0])
            y[1:] = y[:3]
            u[1:] = u[:2]
            
        else:
            u[0] = float(np.sin(np.pi * i / 45.0))
            y[0] = evaluate_plant(y[1:], u[1:])
            mnn.feedforward([y[1], u[0]])
            mnn.backprop(y[0])
            y[1:] = y[:3]
            u[1:] = u[:2]

        print("Iteration: %5d, Loss (squared error): %5.7f" % (i + 1, mnn.squared_error))
    
    if not is_unstable:   
        print("\n Training Finished!, model saved in the folder trained_models/")
        model = open("trained_models/trained_mnn.obj", "wb")
        pickle.dump(mnn, model)
        model.close()

if __name__ == "__main__":
    main()
