import streamlit as st 
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Streamlit page configuration
st.set_page_config(
    page_title = "ReLU Neural Network ",
    layout = "wide"
)

st.title("Neural Network Classification with ReLU Activation")
st.write(
    """
    This app trains a feed-forward neural network using PyTorch with ReLU activation fucntion in hidden layers on synthetic 2D datasets. Features are properly standardized (fit on train only). Visualizations: 
    - Activation curve
    - Dataset
    - Loss curve 
    - Accuracy
    - Decision boundary
    """
)

#Sidebar controls
st.sidebar.header("Settings")

dataset_type = st.sidebar.selectbox("Dataset type",["Moons", "Circles"])
n_samples = st.sidebar.slider("Number of samples", 200, 2000, 500, step=100)
noise = st.sidebar.slider("Noise level", 0.0, 0.5, 0.2, step=0.05)

test_size = st.sidebar.slider("Test size (fraction)", 0.1, 0.5, 0.3, step=0.05)

hidden_size = st.sidebar.slider("Hidden layer size", 4, 128, 16, step=4)
n_hidden_layers = st.sidebar.slider("Number of hidden layers", 1, 3, 2)

learning_rate = st.sidebar.select_slider(
    "Learning rate",
    options=[0.001, 0.003, 0.01, 0.03, 0.1], 
    value=0.01)

epochs = st.sidebar.slider("Epochs", 10, 500, 100, step=10)
batch_size = st.sidebar.slider("Batch size", 16, 256, 64, step=16)

seed = st.sidebar.number_input("Random seed", 0, 9999, 42)

device = torch.device("cpu")

#Generate data
st.subheader("1. Generated Dataset")

torch.manual_seed(seed)
np.random.seed(seed)

if dataset_type == "Moons":
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
else:
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=seed)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

#Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_scaled = scaler.transform(X)

col1, col2 = st.columns(2)
with col1:
    st.write("Sample of training data (first 5 rows scaled):")
    st.dataframe(
        {"x1": X_train_scaled[:5,0], 
         "x2": X_train_scaled[:5,1], 
         "y": y_train[:5]})
    
with col2:
    fig_data, ax_data = plt.subplots()
    ax_data.scatter(X_scaled[:,0], X_scaled[:,1], c=y)
    ax_data.set_xlabel("x1")
    ax_data.set_ylabel("x2")
    ax_data.set_title("Dataset (colored by class)")
    st.pyplot(fig_data)

#PyTorch dataset and dataloader
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
train_dataset = SimpleDataset(X_train_scaled, y_train)
test_dataset = SimpleDataset(X_test_scaled, y_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

input_dim = X_train_scaled.shape[1]
output_dim = len(np.unique(y))

#Neural network model with ReLU 
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_hidden_layers=1):
        super().__init__()
        layers = []

        #First hidden layers
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        #Additional hidden layers
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        #Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model = MLP(input_dim, hidden_size, output_dim, n_hidden_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#Training function
def train_model(model, train_loader, criterion, optimizer, device, epochs):
    model.train()
    loss_history = []

    for epoch in range(epochs):
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_X.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        loss_history.append(epoch_loss)

    return loss_history 

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    return correct / total

#Train model
st.subheader("2. Train the Neural Network")

if st.button("Train Model"):
    with st.spinner("Training..."):
        loss_history = train_model(
            model, train_loader, criterion, optimizer, device, epochs)
        train_acc = evaluate_model(model, train_loader, device)
        test_acc = evaluate_model(model, test_loader, device)
    
    col_loss, col_metrics = st.columns(2)

    with col_loss:
        st.write("Training Loss Curve:")
        fig_loss, ax_loss = plt.subplots()
        ax_loss.plot(range(1, epochs + 1), loss_history)
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        st.pyplot(fig_loss)

    with col_metrics:
        st.write("Accuracy")
        st.metric("Train Accuracy", f"{train_acc * 100:.2f}%")
        st.metric("Test Accuracy", f"{test_acc * 100:.2f}%")

    #Decision boundary visualization
    st.subheader("3. Decision Boundary")

    #Create a mesh grid
    x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
    y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200))
    
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.from_numpy(grid).float().to(device)

    model.eval()
    with torch.no_grad():
        logits = model(grid_tensor)
        _, preds = torch.max(logits, 1)
    Z = preds.cpu().numpy().reshape(xx.shape)

    fig_dec, ax_dec = plt.subplots()
    ax_dec.contourf(xx, yy, Z, alpha=0.4)
    ax_dec.scatter(X_scaled[:,0], X_scaled[:,1], c=y, edgecolor='k')
    ax_dec.set_xlabel("x1")
    ax_dec.set_ylabel("x2")
    ax_dec.set_title("Decision Boundary with ReLU")
    st.pyplot(fig_dec)

else:
    st.write("Click the 'Train Model' button to start training.")