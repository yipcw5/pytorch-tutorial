# Training, testing, saving of neural_networks.py

import torch
import torchvision
import torchvision.transforms as transforms

#1. Loading

num_epochs = 10
batch_size = 32

#check if pytorch installed w/ CUDA support and therefore uses GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#MNIST db contains handwritten digits
#now, retrieve MNIST dataset and put train/test sets into separate tensors
train_dataset = torchvision.datasets.MNIST(root='data',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='data',
                                          train=False,
                                          transform=transforms.ToTensor())

#pass dataset to a torch dataloader => prep data to pass to model w/ spec batch_size, optional shuffling
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
                                    
#2. Training

#alter weights, learning rate, etc. => reduce loss fn => reduce losses
optimiser = torch.optim.Adam(torch.model.parameters(), lr = 0.001)

#calc gradients and update weights // calc how good a prediction model is
loss_function = torch.nn.CrossEntropyLoss()

#explicity transfer all network models and datasets from CPU to GPU; do same for img data later
torch.model.to(device)

'''
Training loops:
1. All loops go thru each epoch and batch in training data loader
2. On each loop iteration, img data and labels transferred to GPU
3. Each loop also applies forward/backward passes and optimisaton steps
4. model applied to imgs in batch, then loss for batch calc
5. gradients calc'ed and back-propagated thru network
'''

total_step = len(train_loader)
##1
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        ##2 Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        ##3 Forward pass, backward pass and optimise
        ##4?
        outputs = torch.model(images)
        loss = loss_function(outputs, labels)

        torch.optimizer.zero_grad()
        loss.backward()
        torch.optimizer.step()

        ##5?
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))