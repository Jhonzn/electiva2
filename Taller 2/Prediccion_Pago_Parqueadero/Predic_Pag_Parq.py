import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
# Cargar dataset
df = pd.read_excel(r"Prediccion_Pago_Parqueadero\parqueadero.xlsx")

# Variables
X = df[["horas", "tipo_vehiculo", "frecuente"]]
y = df["pago_alto"]

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo
modelo = LogisticRegression()
modelo.fit(X_train, y_train)

# Evaluación
y_pred = modelo.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReporte:")
print(classification_report(y_test, y_pred))

# Predicción con ingreso de datos
print("\n--- Nueva predicción ---")
horas = float(input("Ingrese horas: "))
tipo = int(input("Tipo de vehículo (1=carro, 2=moto): "))
frecuente = int(input("Cliente frecuente (0=no, 1=si): "))

nuevo = [[horas, tipo, frecuente]]

pred = modelo.predict(nuevo)[0]
prob = modelo.predict_proba(nuevo)[0][1]

print("\nResultado:")
print("Pago alto:", pred)
print("Probabilidad:", round(prob, 2))
