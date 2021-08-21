# MemoryNeuralNetwork
This repository contains a basic Python implementation of the Memory Neural Network, as proposed in [1]. The program is implemented in using numpy library.

# Install the libraries
```
pip3 install numpy matplotlib --user
```
# Clone the repository
```
git clone https://github.com/NishanthARao/MemoryNeuralNetwork.git
```
# Train the Network
The program implements Example 3.1 as described in [1].
```
cd MemoryNeuralNetwork
python3 train.py
```
The trained model is saved in the folder `trained_models`.
# Test the network
The first reference trajectory (as shown in Fig.3 of [1]) is chosen.
```
cd MemoryNeuralNetwork
python3 test.py
```

[1] P. S. Sastry, G. Santharam and K. P. Unnikrishnan, "Memory neuron networks for identification and control of dynamical systems," in IEEE Transactions on Neural Networks, vol. 5, no. 2, pp. 306-319, March 1994, doi: 10.1109/72.279193. [Available Online](https://ieeexplore.ieee.org/abstract/document/279193).
