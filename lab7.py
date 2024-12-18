import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import os

# Verificar si los archivos existen
if not os.path.exists("C:\\Users\\crist\\Downloads\\iris (1)\\bezdekIris.csv"):
    print("El archivo Iris no se encuentra.")
if not os.path.exists("C:\\Users\\crist\\Downloads\\breast+cancer+wisconsin+diagnostic\\wdbc.csv"):
    print("El archivo WDBC no se encuentra.")
if not os.path.exists("C:\\Users\\crist\\Downloads\\wine+quality (1)\\winequality-red.csv"):
    print("El archivo Wine Quality no se encuentra.")

# Cargar los tres datasets
iris_df = pd.read_csv("C:\\Users\\crist\\Downloads\\iris (1)\\bezdekIris.csv")
wdbc_df = pd.read_csv("C:\\Users\\crist\\Downloads\\breast+cancer+wisconsin+diagnostic\\wdbc.csv")
wine_df = pd.read_csv("C:\\Users\\crist\\Downloads\\wine+quality (1)\\winequality-red.csv", delimiter=";")

# Mostrar las primeras filas de cada dataset para revisar las columnas
print("Iris Dataset:")
print(iris_df.head())

print("\nWDBC Dataset:")
print(wdbc_df.head())

print("\nWine Quality Dataset:")
print(wine_df.head())

# Corregir la carga del dataset WDBC para darle nombres a las columnas
wdbc_df.columns = ['id', 'diagnosis'] + [f'feature_{i}' for i in range(1, len(wdbc_df.columns)-1)]

# Preprocesamiento para cada dataset
def preprocess_data(df, target_column):
    # Dividir en características (X) y etiquetas (y)
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return X, y

# Iris dataset - 'Class' es la columna objetivo
X_iris, y_iris = preprocess_data(iris_df, 'Class')

# WDBC dataset - 'diagnosis' es la columna objetivo
X_wdbc, y_wdbc = preprocess_data(wdbc_df, 'diagnosis')

# Wine dataset - 'quality' es la columna objetivo
X_wine, y_wine = preprocess_data(wine_df, 'quality')

# Función para aplicar KNN y obtener métricas
def knn_evaluation(X, y, k_values=[3, 5, 7]):
    results = {}
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        
        # Hold-Out 70/30 estratificado
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        results[k] = {
            'Accuracy': accuracy,
            'Confusion Matrix': cm
        }
        
        # 10-Fold Cross-Validation estratificado
        strat_kfold = StratifiedKFold(n_splits=10)
        cv_accuracy = cross_val_score(knn, X, y, cv=strat_kfold, scoring='accuracy').mean()
        
        results[k]['10-Fold CV Accuracy'] = cv_accuracy
        
        # Leave-One-Out
        loo = LeaveOneOut()
        loo_accuracy = cross_val_score(knn, X, y, cv=loo, scoring='accuracy').mean()
        results[k]['Leave-One-Out Accuracy'] = loo_accuracy
    
    return results

# Evaluar con los tres datasets
results_iris = knn_evaluation(X_iris, y_iris)
results_wdbc = knn_evaluation(X_wdbc, y_wdbc)
results_wine = knn_evaluation(X_wine, y_wine)

# Mostrar los resultados para cada dataset
print("Iris Results:")
print(results_iris)

print("\nWDBC Results:")
print(results_wdbc)

print("\nWine Results:")
print(results_wine)
