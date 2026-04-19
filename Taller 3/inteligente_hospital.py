import numpy as np
import pandas as pd

# -----------------------------
# CARGAR DATASET DESDE CSV
# -----------------------------
data = pd.read_csv("hospital_dataset.csv", sep=",")

X = data[["gravedad", "habitacion_disponible", "tipo_habitacion"]].values
y = data["autorizado"].values


# Normalizar gravedad paciente
X[:,0] = X[:,0] / 100


# -----------------------------
# PERCEPTRÓN
# -----------------------------
class Perceptron:
    def __init__(self, lr=0.1, epochs=20):
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


# -----------------------------
# VALIDACIONES
# -----------------------------
def validar_gravedad_paciente(valor):
    return 0 <= valor <= 100

def validar_binario(valor):
    return valor in [0, 1]


# -----------------------------
# ENTRENAR MODELO
# -----------------------------
modelo = Perceptron()
modelo.train(X, y)


# -----------------------------
# SISTEMA INTERACTIVO
# -----------------------------
def sistema():
    try:
        gravedad = float(input("Ingrese gravedad de paciente (%): "))
        if not validar_gravedad_paciente(gravedad):
            print("Error: Gravedad del paciente debe estar entre 0 y 100")
            return

        diponible = int(input("¿La habitación esta disponible? (1/0): "))
        if not validar_binario(diponible):
            print("Error: Disponibilidad solo puede ser 0 o 1")
            return

        tipo_habitacion = int(input("¿Requiere habitación tipo UCI? (1/0): "))
        if not validar_binario(tipo_habitacion):
            print("Error: Tipo de habitación solo puede ser 0 o 1")
            return

        entrada = np.array([gravedad / 100, diponible, tipo_habitacion])
        resultado = modelo.predict(entrada)

        print("\nResultado:")
        if resultado == 1:
            print("Asignación autorizada (el paciente puede ingresar)")
        else:
            print("No autorizado (debe esperar o ser remitido)")

    except:
        print("Error: Entrada inválida")


# -----------------------------
# EJECUTAR
# -----------------------------
sistema()