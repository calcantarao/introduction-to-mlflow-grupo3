# Importación de librerías
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Utiliza el algoritmo Random Forest en el entrenamiento del modelo. 
def train_model(X_train, y_train, X_test, y_test):
    clf = RandomForestClassifier(n_estimators=50)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return clf, accuracy