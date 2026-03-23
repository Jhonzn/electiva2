import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, classification_report

# 1. Leer archivo Excel
datos = pd.read_csv("aprobacion.csv")


print("Dataset cargado:")
print(datos)

# 2. Variables independientes
X = datos[["asistencia", "promedio", "participacion"]]

# 3. Variable dependiente
y = datos["aprueba"]

# 4. Dividir dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. Crear modelo
modelo = LinearRegression()

# 6. Entrenar modelo
modelo.fit(X_train, y_train)

# 7. Predicciones del modelo
predicciones_continuas = modelo.predict(X_test)

# 8. Convertir a clasificación
predicciones = []

for p in predicciones_continuas:
    if p >= 0.5:
        predicciones.append(1)
    else:
        predicciones.append(0)

# 9. Matriz de confusión
matriz = confusion_matrix(y_test, predicciones)

print("\nMatriz de Confusión:")
print(matriz)

# 10. Extraer valores
TN = matriz[0][0]
FP = matriz[0][1]
FN = matriz[1][0]
TP = matriz[1][1]

print("\nInterpretación de la matriz de confusión")
print("---------------------------------------")
print("Verdaderos Negativos (TN):", TN)
print("Falsos Positivos (FP):", FP)
print("Falsos Negativos (FN):", FN)
print("Verdaderos Positivos (TP):", TP)

# 11. Métricas del modelo
print("\nReporte de Clasificación:")
print(classification_report(y_test, predicciones))

# 12. Predicción con datos del usuario
print("\n--- Predicción de si un estudiante aprueba ---")

tamano = float(input("Ingrese la asistencia: "))
habitaciones = int(input("Ingrese el promedio: "))
edad = int(input("Ingrese la participación: "))


nuevo_dato = pd.DataFrame(
    [[tamano, habitaciones, edad]],
    columns=X.columns
)

pred = modelo.predict(nuevo_dato)

if pred[0] >= 0.5:
    resultado = 1
else:
    resultado = 0

print("\nResultado de la predicción:")

if resultado == 1:
    print("El estudiante probablemente aprobara")
else:
    print("El estudiante probablemente no aprobara")