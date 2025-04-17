import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import joblib

# ---------------------- Load Dataset ---------------------- #
file_path = "Blublublu.csv"
df = pd.read_csv(file_path)
df['Sy'] = pd.to_numeric(df['Sy'], errors='coerce')
df.dropna(inplace=True)

X = df[['Ro']].values
y = df.drop(columns=['Ro']).values

# Scaling
scaler_X = RobustScaler()
scaler_y = RobustScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

joblib.dump(scaler_X, "scaler_X.pkl")
joblib.dump(scaler_y, "scaler_y.pkl")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------- Quantum Model ---------------------- #
num_qubits = min(4, y.shape[1])
dev = qml.device("default.qubit", wires=num_qubits)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(num_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(num_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

num_layers = 2
weight_shapes = {"weights": (num_layers, num_qubits, 3)}
qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

class QuantumPerceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 16)
        self.fc2 = nn.Linear(16, num_qubits)
        self.q_layer = qlayer
        self.fc3 = nn.Linear(num_qubits, 16)
        self.fc4 = nn.Linear(16, y.shape[1])

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.q_layer(x)
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

model = QuantumPerceptron().to(device)

# ---------------------- Dynamic Training Without Epochs ---------------------- #
loss_weights = torch.tensor([5, 3, 1, 1, 1, 1, 1], dtype=torch.float32).to(device)

def weighted_mse_loss(pred, target):
    loss = ((pred - target) ** 2) * loss_weights
    return loss.mean()

criterion = weighted_mse_loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

batch_size = 64
target_loss = 0.0005
patience_limit = 20

def train_until_convergence(model, X_train, y_train):
    dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                             torch.tensor(y_train, dtype=torch.float32))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_loss = float('inf')
    patience_counter = 0

    while True:
        model.train()
        total_loss = 0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                
                break

        if avg_loss <= target_loss:
            
            break

train_until_convergence(model, X_train, y_train)

# ---------------------- Evaluation ---------------------- #
def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        predictions = model(X_test_tensor).cpu().numpy()

    predictions = scaler_y.inverse_transform(predictions)
    y_test_real = scaler_y.inverse_transform(y_test)

    mse = ((predictions - y_test_real) ** 2).mean()
    print(f"Test MSE: {mse:.4f}")

evaluate(model, X_test, y_test)

torch.save(model.state_dict(), "quantum_perceptron_fast.pth")

# ---------------------- Load Model ---------------------- #
def load_model():
    model = QuantumPerceptron().to(device)
    model.load_state_dict(torch.load("quantum_perceptron_fast.pth"))
    model.eval()
    return model

# ---------------------- Prediction (Formatted Output) ---------------------- #
def predict_properties(mass_density):
    model = load_model()
    mass_density_scaled = scaler_X.transform([[mass_density]])
    mass_density_tensor = torch.tensor(mass_density_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        prediction = model(mass_density_tensor).cpu().numpy()

    prediction = scaler_y.inverse_transform(prediction)
    property_names = df.columns.drop("Ro")

    print("\nThe predicted Properties are:")
    for name, value in zip(property_names, prediction[0]):
        print(f"{name}: {value:.4f}")

    return prediction

# ---------------------- User Input ---------------------- #
user_input = float(input("Enter mass density: "))
predict_properties(user_input)