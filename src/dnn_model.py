import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes, output_dim=1, activation='relu', dropout_rate=0.3, use_batchnorm=True):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_dnn(X_train, y_train, X_val=None, y_val=None, config=None, return_history=False):
    if config is None:
        hidden_sizes = [128, 64, 32]
        batch_size = 64
        learning_rate = 0.001
        n_epochs = 100
        dropout_rate = 0.3
        use_batchnorm = True
        activation = 'relu'
    else:
        hidden_sizes = config.dnn_hidden_sizes
        batch_size = config.dnn_batch_size
        learning_rate = config.dnn_learning_rate
        n_epochs = config.dnn_n_epochs
        dropout_rate = config.dnn_dropout_rate
        use_batchnorm = config.dnn_batch_norm
        activation = config.dnn_activation

    if hasattr(X_train, 'values'):
        X_train = X_train.values
    if hasattr(y_train, 'values'):
        y_train = y_train.values

    X_t = torch.FloatTensor(X_train)
    y_t = torch.FloatTensor(y_train).reshape(-1, 1)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    val_loader = None
    if X_val is not None and y_val is not None:
        if hasattr(X_val, 'values'):
            X_val = X_val.values
        if hasattr(y_val, 'values'):
            y_val = y_val.values
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val).reshape(-1, 1)
        val_dataset = TensorDataset(X_val_t, y_val_t)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = MLP(input_dim=X_train.shape[1], hidden_sizes=hidden_sizes, output_dim=1, activation=activation, dropout_rate=dropout_rate, use_batchnorm=use_batchnorm)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()   # для регрессии

    history = {'train_loss': [], 'val_loss': [], 'val_rmse': []}
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader)
        history['train_loss'].append(avg_loss)

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            preds_all = []
            y_all = []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    preds = model(batch_X)
                    loss = criterion(preds, batch_y)
                    val_loss += loss.item()
                    preds_all.append(preds.numpy())
                    y_all.append(batch_y.numpy())
            avg_val_loss = val_loss / len(val_loader)
            y_all = np.concatenate(y_all)
            preds_all = np.concatenate(preds_all)
            rmse = np.sqrt(np.mean((y_all - preds_all)**2))
            history['val_loss'].append(avg_val_loss)
            history['val_rmse'].append(rmse)
            model.train()

        if (epoch+1) % 20 == 0:
            if val_loader is not None:
                print(f"  DNN epoch {epoch+1}/{n_epochs}, loss: {avg_loss:.4f}, val_loss: {avg_val_loss:.4f}, val_rmse: {rmse:.4f}")
            else:
                print(f"  DNN epoch {epoch+1}/{n_epochs}, loss: {avg_loss:.4f}")

    model.eval()
    if return_history:
        return model, history
    return model

def predict_dnn(model, X):
    model.eval()
    if hasattr(X, 'values'):
        X = X.values
    with torch.no_grad():
        X_t = torch.FloatTensor(X)
        preds = model(X_t).numpy().flatten()
    return preds