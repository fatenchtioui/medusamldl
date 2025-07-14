# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import pickle
# from sklearn.metrics import accuracy_score, precision_score
# class MatchingANN(nn.Module):
#     def __init__(self, input_dim):
#         super(MatchingANN, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 256)
#         self.fc2 = nn.Linear(256, 2)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         return self.fc2(x)

# class MatchingDLTrainer:
#     def __init__(self, input_dim):
#         self.input_dim = input_dim
#         self.model = MatchingANN(input_dim)
#         self.trained = False

#     def train(self, X_train, y_train, epochs=10, lr=1e-3):
#         X_train = torch.tensor(X_train, dtype=torch.float32)
#         y_train = torch.tensor(y_train, dtype=torch.long)
#         dataset = torch.utils.data.TensorDataset(X_train, y_train)
#         loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
#         optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
#         loss_fn = nn.CrossEntropyLoss()

#         for epoch in range(epochs):
#             for xb, yb in loader:
#                 optimizer.zero_grad()
#                 pred = self.model(xb)
#                 loss = loss_fn(pred, yb)
#                 loss.backward()
#                 optimizer.step()
#         self.trained = True
#         return self.evaluate(X_train, y_train)

#     def evaluate(self, X, y):
#         self.model.eval()
#         with torch.no_grad():
#             X = torch.tensor(X, dtype=torch.float32)
#             y = torch.tensor(y, dtype=torch.long)
#             outputs = self.model(X)
#             preds = torch.argmax(outputs, dim=1)
#             accuracy = (preds == y).float().mean().item()
#         self.model.train()
#         return {"accuracy": accuracy}

#     def predict_proba(self, X):
#         self.model.eval()
#         with torch.no_grad():
#             X = torch.tensor(X, dtype=torch.float32)
#             outputs = self.model(X)
#             probs = torch.softmax(outputs, dim=1)[:, 1]  # Probabilité classe 1
#         self.model.train()
#         return probs.numpy()

#     def export_model(self, vectorizer):
#         return pickle.dumps({
#             'model_state_dict': self.model.state_dict(),
#             'input_dim': self.input_dim,
#             'vectorizer': vectorizer  # Sauvegarde du vectorizer
#         })

#     def load_model(self, state_bytes):
#         data = pickle.loads(state_bytes)
#         self.model = MatchingANN(data['input_dim'])
#         self.model.load_state_dict(data['model_state_dict'])
#         self.trained = True
#         return data['vectorizer']  # Retourne le vectorizer chargé
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score

class ANNModel(nn.Module):
    """Réseau de neurones standard (1 couche cachée)"""
    def __init__(self, input_dim):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class MLPModel(nn.Module):
    """Réseau multicouche (2 couches cachées)"""
    def __init__(self, input_dim):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DNNModel(nn.Module):
    """Réseau profond (3+ couches cachées)"""
    def __init__(self, input_dim):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        return self.fc5(x)

class MatchingDLTrainer:
    def __init__(self, input_dim, model_type="ann"):
        self.input_dim = input_dim
        self.model = self._init_model(model_type)
        self.trained = False
        self.model_type = model_type

    def _init_model(self, model_type):
        if model_type == "mlp":
            return MLPModel(self.input_dim)
        elif model_type == "dnn":
            return DNNModel(self.input_dim)
        else:  # "ann" par défaut
            return ANNModel(self.input_dim)

    def train(self, X_train, y_train, epochs=10, lr=1e-3):
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()
        self.trained = True
        return self.evaluate(X_train, y_train)

    def evaluate(self, X, y):
        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.long)
            outputs = self.model(X)
            preds = torch.argmax(outputs, dim=1)
            accuracy = (preds == y).float().mean().item()
        self.model.train()
        return {"accuracy": accuracy}

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32)
            outputs = self.model(X)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Probabilité classe 1
        self.model.train()
        return probs.numpy()

    def export_model(self, vectorizer):
        return pickle.dumps({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.input_dim,
            'vectorizer': vectorizer,
            'model_type': self.model_type  # Sauvegarde du type de modèle
        })

    def load_model(self, state_bytes):
        data = pickle.loads(state_bytes)
        self.model = self._init_model(data.get('model_type', 'ann'))
        self.model.load_state_dict(data['model_state_dict'])
        self.input_dim = data['input_dim']
        self.trained = True
        self.model_type = data.get('model_type', 'ann')
        return data['vectorizer']