import torch 
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader, TensorDataset
from  keras.datasets import mnist

# Create a class that inherits nn.Module
class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super.__init__()
        self.fc1 = nn.linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.out(x)
        return x
    
# Training function
def train_model(model, train_loader, criterion, optimizer, epochs):
    # Set the model in train mode
    model.train()
    for _ in range(epochs):
        #  Get the batches
        for images, labels in train_loader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        

def main():
    # Load the MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # Normalize the data between 0 and 1 (only for the images)
    X_train, X_test = X_train / 255.0, X_test / 255.0
    
    # Convert the data from numpy arrays to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    # Add a channel dimension to the images (for compatibility with convolutional layers)
    #X_train = X_train.unsqueeze(1)  # Shape: (60000, 1, 28, 28)
    #X_test = X_test.unsqueeze(1)    # Shape: (10000, 1, 28, 28)
    
    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)
    
    
    
if __name__ == "__main__":
    main()