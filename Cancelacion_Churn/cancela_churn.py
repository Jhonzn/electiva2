# MODELO USANDO DATASET PERSONALIZADO
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Cargar dataset

data = pd.read_excel("dataset_clientes_churn.xlsx")

# Separar variables
X = data.drop("cancelo", axis=1)
y = data["cancelo"]

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

print("\nIngrese los datos del cliente:")

compras_previas = float(input("Número de compras previas: "))
tiempo_como_cliente_meses = float(input("Tiempo como cliente (meses): "))
valor_pedido = float(input("Valor promedio del pedido: "))
cancelaciones_previas = float(input("Cancelaciones previas: "))
tiempo_promedio_entre_compras_dias = float(input("Tiempo promedio entre compras (días): "))

nuevo_cliente = pd.DataFrame(
    [[compras_previas,
      tiempo_como_cliente_meses,
      valor_pedido,
      cancelaciones_previas,
      tiempo_promedio_entre_compras_dias]],
    columns=X.columns
)

prediccion = modelo.predict(nuevo_cliente)

print("\nResultado de la predicción:")
print("¿El cliente cancelará?",
      "Sí" if prediccion[0] == 1 else "No")


