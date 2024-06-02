import numpy as np
from torchvision import datasets, transforms

def load_mnist():
    # Cargando el conjunto de datos MNIST
    mnist_train = datasets.MNIST(root='./', train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST(root='./', train=False, download=True, transform=transforms.ToTensor())
    # Convertimos los datos a NumPy arrays
    train_images = mnist_train.data.numpy()
    train_labels = mnist_train.targets.numpy()
    test_images = mnist_test.data.numpy()
    test_labels = mnist_test.targets.numpy()
    return train_images, train_labels, test_images, test_labels

def initialize_filters(size, scale=1.0):
    stddev = scale / np.sqrt(np.prod(size))
    return np.random.normal(loc=0, scale=stddev, size=size)

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

def softmax(x):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

# Main/ Ejemplo de uso
# Se descarga el conjunto de datos MNIST y se transforma en Numpy array
train_images, train_labels, test_images, test_labels = load_mnist()

# Inicializar filtros y pesos
f1 = initialize_filters([8, 1, 3, 3])

for i in range(len(train_images)):  # Procesa cada imagen
    image = train_images[i] / 255.0
    image = image.reshape(1, 28, 28)

    # Feedforward
    conv_out = convolution(image, f1, np.zeros((8, 1)))
    relu_out = np.maximum(conv_out, 0)
    pool_out = maxpool(relu_out)
    flattened = pool_out.flatten()

    # Ajusta la segunda dimensión de acuerdo a las características aplanadas si es necesario
    if i == 0:
        flattened_length = flattened.shape[0]
        w1 = np.random.randn(10, flattened_length)  # Inicializa w1 solo una vez

    dense_out = softmax(np.dot(w1, flattened))
    print(f"Salida de la red para la imagen {i}:", dense_out)
    predicted_class = np.argmax(dense_out)
    print(f"Clasificacion de la red para la imagen {i}, valor real: {train_labels[i]}, valor predicho", predicted_class)

