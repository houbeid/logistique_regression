import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def sigmoid(z):
    z = np.asarray(z)
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def log_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def train_logistic_regression(X, y, learning_rate=0.01, max_iterations=10000, tolerance=1e-7):
    """Entraîne un modèle de régression logistique"""
    weights = np.zeros(X.shape[1])
    loss = float('inf')
    
    for iteration in range(max_iterations):
        z = X @ weights
        predictions = sigmoid(z)
        
        error = predictions - y
        gradient = X.T @ error / len(y)
        
        weights -= learning_rate * gradient
        
        new_loss = log_loss(y, predictions)
        if abs(new_loss - loss) < tolerance:
            print(f"  Convergence à l'itération {iteration}")
            break
        loss = new_loss
    return weights


df = pd.read_csv("datasets/dataset_train.csv")
df.drop(columns=['Index', 'First Name', 'Last Name', 'Birthday'], inplace=True)
houses = df['Hogwarts House'].copy()
df = pd.get_dummies(df, drop_first=True)
similar_features = [
    ('Astronomy', 'Defense Against the Dark Arts'),
    ('Flying', 'History of Magic')
]

for f1, f2 in similar_features:
    nan_f1 = df[f1].isna().sum()
    nan_f2 = df[f2].isna().sum()

    if nan_f1 > nan_f2:
        df.drop(columns=[f1], inplace=True)
    else:
        df.drop(columns=[f2], inplace=True)
print(df.columns)
df.fillna(df.mean(), inplace=True)
house_cols = ['Hogwarts House_Ravenclaw', 'Hogwarts House_Slytherin', 'Hogwarts House_Hufflepuff']
numeric_cols = df.select_dtypes(include='number').columns.difference(house_cols)
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
X = df.drop(house_cols, axis=1).to_numpy(dtype=float)
X = np.hstack((np.ones((X.shape[0], 1)), X))
y_Gryffindor = (houses == 'Gryffindor').astype(int).values
y_Slytherin = (houses == 'Slytherin').astype(int).values
y_Ravenclaw = (houses == 'Ravenclaw').astype(int).values
y_Hufflepuff = (houses == 'Hufflepuff').astype(int).values
weights_gryffindor = train_logistic_regression(X, y_Gryffindor)
weights_slytherin = train_logistic_regression(X, y_Slytherin)
weights_ravenclaw = train_logistic_regression(X, y_Ravenclaw)
weights_hufflepuff = train_logistic_regression(X, y_Hufflepuff)
np.savetxt('weights_gryffindor.txt', weights_gryffindor)
np.savetxt('weights_slytherin.txt', weights_slytherin)
np.savetxt('weights_ravenclaw.txt', weights_ravenclaw)
np.savetxt('weights_hufflepuff.txt', weights_hufflepuff)