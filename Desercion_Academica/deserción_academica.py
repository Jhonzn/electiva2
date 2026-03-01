# MODELO USANDO DATASET PERSONALIZADO
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Cargar dataset

data = pd.read_excel("dataset_academico_estudiantes.xlsx")

# Separar variables
X = data.drop("deserto", axis=1)
y = data["deserto"]

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Crear modelo
modelo = DecisionTreeClassifier()

# Entrenar
modelo.fit(X_train, y_train)

# Evaluar
y_pred = modelo.predict(X_test)
print("Exactitud del modelo:", accuracy_score(y_test, y_pred))

# Entrada de datos por teclado

print("\nIngrese los datos del estudiante:")

horas_estudio_semana = float(input("Horas de estudio semana: "))
inasistencias = float(input("Inasistencia: "))
promedio_previo = float(input("Promedio previo: "))
tiempo_plataforma_horas = float(input("Tiempo en plataforma Hrs: "))
ingresos_familiares = float(input("Ingresos familiares: "))
distancia_km = float(input("Distancia domicilio Km: "))
uso_tutorias = float(input("Uso de tutorias: "))
prestamos_biblioteca = float(input("Prestamos de libros: "))

nuevo_estudiante = pd.DataFrame(
    [[horas_estudio_semana, inasistencias, promedio_previo, 
      tiempo_plataforma_horas, ingresos_familiares, 
      distancia_km, uso_tutorias, prestamos_biblioteca]],
    columns=X.columns
)

prediccion = modelo.predict(nuevo_estudiante)

print("\nResultado de la predicción:")
print("¿Deserta?", "Sí" if prediccion[0] == 1 else "No")