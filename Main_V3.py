import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


# Función para cargar MNIST usando PyTorch
def load_mnist():
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return train_dataset, test_dataset

def initialize_filters(size, scale=1.0):
    stddev = scale / np.sqrt(np.prod(size))
    return np.random.normal(loc=0, scale=stddev, size=size)

def maxpool(image, f=2, s=2):
    n_c, h_prev, w_prev = image.shape
    h = int((h_prev - f) / s) + 1
    w = int((w_prev - f) / s) + 1
    downsampled = np.zeros((n_c, h, w))
    for i in range(n_c):
        curr_y = out_y = 0
        while curr_y + f <= h_prev:
            curr_x = out_x = 0
            while curr_x + f <= w_prev:
                downsampled[i, out_y, out_x] = np.max(image[i, curr_y:curr_y + f, curr_x:curr_x + f])
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
    return downsampled

def convolution(image, filt, bias, s=1):
    n_f, n_c_f, f, _ = filt.shape
    n_c, in_dim, _ = image.shape

    out_dim = int((in_dim - f) / s) + 1
    assert n_c == n_c_f, "Dimensiones de canal deben coincidir."
    output = np.zeros((n_f, out_dim, out_dim))
    for curr_f in range(n_f):
        curr_y = out_y = 0
        while curr_y + f <= in_dim:
            curr_x = out_x = 0
            while curr_x + f <= in_dim:
                output[curr_f, out_y, out_x] = np.sum(filt[curr_f] * image[:, curr_y:curr_y + f, curr_x:curr_x + f]) + bias[curr_f]
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
    return output

# Definición de la clase del modelo de Perceptrón Multicapa
class SimpleMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleMLP, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)  # Sin softmax aquí, porque nn.CrossEntropyLoss ya lo incluye


# Carga de datos
train_dataset, test_dataset = load_mnist()

# Inicialización de filtros y bias para la convolución
f1 = initialize_filters([8, 1, 3, 3])
b1 = np.zeros((8, 1))

# Configuración de PyTorch para el modelo y el entrenamiento
input_size = 128  # Asumiendo que esta es la dimensión después de aplanar la salida del pool
num_classes = 10
model = SimpleMLP(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Ciclo de entrenamiento
for images, labels in train_dataset:
    images = images / 255.0 # Normalizar imagenes

    # Procesamiento convolucional con NumPy
    conv_out = np.array([convolution(image.reshape(1, 28, 28), f1, b1) for image in images])

    relu_out = np.maximum(conv_out, 0)
    pool_out = np.array([maxpool(img) for img in relu_out])
    flattened = pool_out.reshape(pool_out.shape[0], -1)

    # Convertir a tensores de PyTorch
    inputs = torch.tensor(flattened).float()
    targets = labels.long()

    # Pasar datos por el modelo de PyTorch
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Retropropagación y optimización
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Loss: {loss.item()}')


