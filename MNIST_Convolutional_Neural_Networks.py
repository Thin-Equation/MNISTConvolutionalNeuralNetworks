import streamlit as st
import torch
import tensorflow as tf
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn, optim
from tensorflow.keras.datasets import mnist

# PyTorch CNN model
class CustomCNN(nn.Module):
    def __init__(self, num_conv_layers, in_channels, out_channels, kernel_size, stride, padding):
        super(CustomCNN, self).__init__()

        self.conv_layers = nn.ModuleList()
        self.num_conv_layers = num_conv_layers

        # Add convolutional layers with BatchNorm, ReLU, and MaxPool
        for _ in range(num_conv_layers):
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            self.conv_layers.append(nn.BatchNorm2d(out_channels))
            self.conv_layers.append(nn.ReLU(inplace=True))
            self.conv_layers.append(nn.MaxPool2d(2))
            in_channels = out_channels

        # Calculate the final output size
        self.calculate_output_size()

        # Add fully connected layer
        self.fc = nn.Linear(self.final_size, 10)  # Assuming 10 output classes

    def forward(self, x):
        # Forward pass through convolutional layers
        for layer in self.conv_layers:
            x = layer(x)

        # Flatten the output
        x = x.view(-1, self.final_size)

        # Forward pass through fully connected layer
        x = self.fc(x)

        return x

    def calculate_output_size(self):
        # Dummy input to calculate the final output size
        x = torch.randn(1, 1, 28, 28)  # Adjusted input size and number of input channels

        # Forward pass through convolutional layers to calculate output size
        for layer in self.conv_layers:
            x = layer(x)

        # Flatten the output
        x = x.view(1, -1)

        # Get the final output size
        self.final_size = x.size(1)

# TensorFlow CNN model
class CustomCNN(tf.keras.Model):
    def __init__(self, num_conv_layers, filters, kernel_size, strides, padding):
        super(CustomCNN, self).__init__()

        self.conv_layers = tf.keras.Sequential()
        for _ in range(num_conv_layers):
            self.conv_layers.add(tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding))
            self.conv_layers.add(tf.keras.layers.BatchNormalization())
            self.conv_layers.add(tf.keras.layers.ReLU())
            self.conv_layers.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.conv_layers(inputs)
        x = self.flatten(x)
        x = self.fc(x)
        return x

def train_pytorch_model(train_loader, model, criterion, optimizer, device):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

def test_pytorch_model(test_loader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def train_tensorflow_model(model, train_images, train_labels, test_images, test_labels, epochs):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels))
    return history

def test_tensorflow_model(model, test_images, test_labels):
    _, accuracy = model.evaluate(test_images, test_labels)
    return accuracy

def main():
    st.title("CNN Models for MNIST Dataset")

    model_choice = st.radio("Select Model Framework", ("PyTorch", "TensorFlow"))

    if model_choice == "PyTorch":
        st.header("PyTorch Model")

        st.sidebar.header("Model Hyperparameters")
        num_conv_layers = st.sidebar.slider("Number of Convolutional Layers", 1, 5, 3)
        out_channel = st.sidebar.slider("Number of Filters", 4, 32, 12)
        kernel_size = st.sidebar.slider("Kernel Size", 3, 7, 4)
        strides = st.sidebar.slider("Stride", 1, 3, 1)
        padding = st.sidebar.selectbox("Padding", [0,1,2])
        
        
        # Downloading and preprocessing the dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
        test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

        # Define the model, optimizer, and criterion
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CustomCNN(num_conv_layers, 1, out_channel, kernel_size, strides, padding).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Training the model
        epochs = 5
        losses = []
        for epoch in range(epochs):
            losses.append(train_pytorch_model(train_loader, model, criterion, optimizer, device))

        # Plotting the epoch loss graph
        st.subheader("Epoch Loss Graph (PyTorch)")
        plt.plot(range(len(losses)), losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        st.pyplot()

        # Testing the model
        accuracy = test_pytorch_model(test_loader, model, device)
        st.write("PyTorch Model Accuracy:", accuracy)

    else:
        st.header("TensorFlow Model")

        st.sidebar.header("Model Hyperparameters")
        num_conv_layers = st.sidebar.slider("Number of Convolutional Layers", 1, 5, 3)
        filters = st.sidebar.slider("Number of Filters", 4, 32, 12)
        kernel_size = st.sidebar.slider("Kernel Size", 3, 7, 4)
        strides = st.sidebar.slider("Stride", 1, 3, 1)
        padding = st.sidebar.selectbox("Padding", ["same", "valid"])
        
        # Downloading and preprocessing the dataset
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)) / 255.0
        test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)) / 255.0
        train_labels = tf.keras.utils.to_categorical(train_labels)
        test_labels = tf.keras.utils.to_categorical(test_labels)

        # Define the model
        model = CustomCNN(num_conv_layers, filters, kernel_size, strides, padding)

        # Training the model
        epochs = 5
        history = train_tensorflow_model(model, train_images, train_labels, test_images, test_labels, epochs)

        # Plotting the epoch loss graph
        st.subheader("Epoch Loss Graph (TensorFlow)")
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        st.pyplot()

        # Testing the model
        accuracy = test_tensorflow_model(model, test_images, test_labels)
        st.write("TensorFlow Model Accuracy:", accuracy)

if __name__ == "__main__":
    main()
