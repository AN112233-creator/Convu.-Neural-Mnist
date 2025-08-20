import torch 
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# MNIST dataset is not available in sklearn, so we will use the digits dataset as a substitute

# Training Function

training_data = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])  )

test_data = torchvision.datasets.MNIST(
    root='./data',  
    train=False,
    download=True,
    transform=transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]) )


# Load the MNIST dataset
training_loader = torch.utils.data.DataLoader(
    dataset=training_data,
    batch_size=64,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_data,
    batch_size=64,
    shuffle=False
)

# Display the first images from the training set
examples = iter(training_loader)
samples, labels = next(examples)
# print(labels)
print(samples.shape, labels.shape)
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(samples[i].squeeze(), cmap='gray')
    plt.title(f"Label: {labels[i]}")
    plt.axis('off')
#plt.show()

# Convulutional Neural Network Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(-1, 64 * 7 * 7)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Initialize the model, loss function, and optimizer
model = CNNModel().to(device)


def train_model():
    # Hyperparameters
    learning_rate = 0.001
    epochs = 5
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    for epoch in range(epochs):
        total_no_steps = len(training_loader)
        model.train()
        for batch, (images, labels) in enumerate(training_loader):
            images = images.to(device)
            labels = labels.to(device) 
            
            # Forward pass
            prediction = model(images)
            loss = criterion(prediction, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
           
            
            if batch % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{batch}/{total_no_steps}], Loss: {loss.item():.4f}')
   

# Evaluation
def evaluate():
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            pred = model(images)
            _, predicted = torch.max(pred.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Accuracy of the model on the test images: {accuracy:.2f}%')
        return accuracy


if __name__ == "__main__":
    train_model()
    evaluate()
    torch.save(model.state_dict(), "mnist_cnn_model.pth")
    print("Model saved as mnist_cnn_model.pth")