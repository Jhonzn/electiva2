import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------------
# CARGAR DATASET
# -----------------------------
data = pd.read_csv("hospital_dataset_parcial3.csv")

X = data[["gravedad", "habitacion_disponible", "tipo_habitacion"]].values
y = data["autorizado"].values

# Normalizar gravedad
X[:, 0] = X[:, 0] / 100


# -----------------------------
# DIVIDIR DATOS
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -----------------------------
# PERCEPTRÓN
# -----------------------------
class Perceptron:
    def __init__(self, lr=0.1, epochs=30):
        self.lr = lr
        self.epochs = epochs

    def train(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.epochs):
            for i in range(len(X)):
                linear = np.dot(X[i], self.weights) + self.bias
                y_pred = 1 if linear >= 0 else 0

                error = y[i] - y_pred
                self.weights += self.lr * error * X[i]
                self.bias += self.lr * error

    def predict(self, x):
        linear = np.dot(x, self.weights) + self.bias
        return 1 if linear >= 0 else 0

    def evaluar(self, X, y):
        correctos = 0
        for i in range(len(X)):
            if self.predict(X[i]) == y[i]:
                correctos += 1
        return correctos / len(X)


# -----------------------------
# ENTRENAR MODELO
# -----------------------------
modelo = Perceptron()
modelo.train(X_train, y_train)

# Evaluación
accuracy = modelo.evaluar(X_test, y_test)
print(f"Precisión del modelo: {accuracy * 100:.2f}%")


# -----------------------------
# VALIDACIONES
# -----------------------------
def validar_gravedad_paciente(valor):
    return 0 <= valor <= 100

def validar_binario(valor):
    return valor in [0, 1]


# -----------------------------
# SISTEMA INTERACTIVO
# -----------------------------
def sistema():
    try:
        gravedad = float(input("Ingrese gravedad de paciente (%): "))
        if not validar_gravedad_paciente(gravedad):
            print("Error: Gravedad debe estar entre 0 y 100")
            return

        disponible = int(input("¿Habitación disponible? (1/0): "))
        if not validar_binario(disponible):
            print("Error: Solo 0 o 1")
            return

        tipo_habitacion = int(input("¿Requiere UCI? (1/0): "))
        if not validar_binario(tipo_habitacion):
            print("Error: Solo 0 o 1")
            return

        entrada = np.array([gravedad / 100, disponible, tipo_habitacion])
        resultado = modelo.predict(entrada)

        print("\nResultado:")
        if resultado == 1:
            print("Asignación autorizada")
        else:
            print("No autorizado")

    except ValueError:
        print("Error: Entrada inválida")


# -----------------------------
# EJECUTAR
# -----------------------------
sistema()
