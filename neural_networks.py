# Neural Networks
## - def as classes, which extends torch.nn.module from torch lib
## Convolutional NN applied on MNIST datatset:

from torch import nn, F

class Net(nn.Module):
    def __init__(self):
        #__init__(): def any network layers

        #2 convolutional layers, w/ conv2 used a few times
        super(Net,self).__init__() 
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3,3), padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3,3), padding=1)
        
        #use pools near end
        #pooling layers enable downsampling - lower-resolution i/p contains key struct elements w/o fine detail that might cause too-sig changes in feature map
        #max: largest value in each patch of each feature map; res = downsampled maps w/ most present (not avg) feature
        #global: aggressively summarise feature presence; res = down samples entire feature map to 1 value, instead of of patches of i/p map
        self.max_pool = nn.MaxPool2d(2,2)
        self.global_pool = nn.AvgPool2d(7)

        #full-connected layers => final o/p probs
        self.fc1 = nn.Linear(64,64)
        self.fc2 = nn.Linear(64,10)

    def forward(self, x):
        #forward(): set up model by stacking layers (conv, pooling, FC)
        #*pytorch can print shape and res of any tensor w/i intermediate layers w/ simple print(), anywhere in forward()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.max_pool(x)

        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = self.max_pool(x)

        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = self.global_pool(x)

        x = x.view(-1, 64)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        #softmax => => final o/p probs
        #last hidden layer before o/p layer, must have same no. of nodes
        x = F.log_softmax(x)

        return x

model = Net()