import torch
import torch.nn as NeuralNetwork
import torch.nn.functional as Functional
from torchvision import transforms
Tensor =  transforms.ToTensor()


class ConvolutionNeuralNetwork(NeuralNetwork.Module):
    def __init__(self):
        super().__init__()
        self.cv1 = NeuralNetwork.Conv2d(1,6,(3,3),1)
        self.cv2 = NeuralNetwork.Conv2d(6,16,(3,3),1)
        dummy_input = torch.randn(1, 1, 28, 28)
        x = Functional.max_pool2d(Functional.relu(self.cv1(dummy_input)), 2, 2)
        x = Functional.max_pool2d(Functional.relu(self.cv2(x)), 2, 2)
        fc1_input_features = torch.flatten(x, 1).size(1)
        self.fc1 = NeuralNetwork.Linear(fc1_input_features,120)
        self.fc2 = NeuralNetwork.Linear(120,64)  
        self.fc3 = NeuralNetwork.Linear(64,10)    
    
    def forward(self,x):
        x = Functional.relu(self.cv1(x))
        x = Functional.max_pool2d(x,2,2)

        x = Functional.relu(self.cv2(x))
        x = Functional.max_pool2d(x,2,2)

        x = torch.flatten(x, 1) 

        x = Functional.relu(self.fc1(x))
        x = Functional.relu(self.fc2(x))
        x = self.fc3(x)
        output = Functional.log_softmax(x,dim=1)
        pred = output.argmax(dim=1, keepdim=True)
        return output, pred