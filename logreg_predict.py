import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def sigmoid(z):
    z = np.asarray(z)
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


weights_gryffindor = np.loadtxt('weights_gryffindor.txt')
weights_slytherin = np.loadtxt('weights_slytherin.txt')
weights_ravenclaw = np.loadtxt('weights_ravenclaw.txt')
weights_hufflepuff = np.loadtxt('weights_hufflepuff.txt')
df_train = pd.read_csv("datasets/dataset_train.csv")
df_train.drop(columns=['Index', 'First Name', 'Last Name', 'Birthday', 'History of Magic', 'Astronomy'], inplace=True)
df_train = pd.get_dummies(df_train, drop_first=True)
df_train.fillna(df_train.mean(), inplace=True)
house_cols = ['Hogwarts House_Ravenclaw', 'Hogwarts House_Slytherin', 'Hogwarts House_Hufflepuff']
feature_cols_train = [col for col in df_train.columns if col not in house_cols]
scaler = StandardScaler()
scaler.fit(df_train[feature_cols_train])
df_test = pd.read_csv("datasets/dataset_test.csv")
student_indices = df_test['Index'].copy()
df_test.drop(columns=['Index', 'First Name', 'Last Name', 'Birthday'], inplace=True)
df_test = pd.get_dummies(df_test, drop_first=True)
df_test.fillna(df_test.mean(), inplace=True)
for col in feature_cols_train:
    if col not in df_test.columns:
        print(f"  Ajout de la colonne manquante: {col}")
        df_test[col] = 0
for col in df_test.columns:
    if col not in feature_cols_train:
        print(f"  Suppression de la colonne en trop: {col}")
        df_test.drop(columns=[col], inplace=True)
df_test = df_test[feature_cols_train]
df_test[feature_cols_train] = scaler.transform(df_test[feature_cols_train])
X = df_test.to_numpy(dtype=float)
X = np.hstack((np.ones((X.shape[0], 1)), X))
if X.shape[1] != len(weights_gryffindor):
    print(f"\n ERREUR: Incompatibilité des dimensions!")
    print(f"  X a {X.shape[1]} colonnes mais les poids en attendent {len(weights_gryffindor)}")
    exit(1)
predictions = []
for i in range(X.shape[0]):
    x = X[i]
    prob_gryffindor = sigmoid(np.dot(x, weights_gryffindor))
    prob_slytherin = sigmoid(np.dot(x, weights_slytherin))
    prob_ravenclaw = sigmoid(np.dot(x, weights_ravenclaw))
    prob_hufflepuff = sigmoid(np.dot(x, weights_hufflepuff))
    
    probabilities = [prob_gryffindor, prob_slytherin, prob_ravenclaw, prob_hufflepuff]
    houses = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']
    predicted_house = houses[np.argmax(probabilities)]
    predictions.append(predicted_house)
results = pd.DataFrame({
    'Index': student_indices,
    'Hogwarts House': predictions
})
results.to_csv("houses.csv", index=False)