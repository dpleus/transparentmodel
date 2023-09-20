import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from transparentmodel.pytorch.training import apply_tracking

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 12800)
        self.fc2 = nn.Linear(12800, 6400)
        self.fc3 = nn.Linear(6400, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.log_softmax(x, dim=1)


# Set random seed for reproducibility
torch.manual_seed(42)

# Define the training parameters
batch_size = 64
epochs = 10
learning_rate = 0.01

# Load the MNIST dataset
train_dataset = datasets.MNIST(
    root="./data", train=True, transform=transforms.ToTensor(), download=True
)

# Create a data loader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)

# Create an instance of the neural network
model = Net().to(device)

# Define the optimizer and loss function
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Train the model
for epoch in range(1, epochs + 1):
    model = apply_tracking(model, realtime=True)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(
                f"Epoch {epoch}/{epochs},"
                f"Batch {batch_idx}/{len(train_loader)},"
                f"Loss: {loss.item()}"
            )

print("Training completed.")
