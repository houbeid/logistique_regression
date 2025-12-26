# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# import numpy as np
# import math



# df = pd.read_csv("datasets/dataset_train.csv")

# # Supprimer les colonnes inutiles
# cols_to_drop = ['Index', 'First Name', 'Last Name', 'Birthday']
# df.drop(columns=cols_to_drop, inplace=True)
# df = pd.get_dummies(df, drop_first=True)
# df.fillna(df.mean(), inplace=True)
# numeric_cols = df.select_dtypes(include='number').columns
# scaler = StandardScaler()
# df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
# y_Ravenclaw = df[['Hogwarts House_Ravenclaw']]
# x = df.drop(['Hogwarts House_Ravenclaw', 'Hogwarts House_Slytherin', 'Hogwarts House_Hufflepuff'], axis=1)
# x = np.hstack((np.ones((x.shape[0], 1)), x))

# num_features = x.shape[1]
# weights = np.zeros(num_features)
# learning_rate = 0.01
# max_iterations = 10000
# m = len(x)
# for iteration in range(max_iterations):
#     weights_temp = [0 * a for a in weights]
#     for i in range(m):
#         z = sum(a * b for a, b in zip(x[0], weights))
#         y = 1 / (1 + math.exp(-z))
#         weights_temp[0] += (y - y_Ravenclaw[i])
#         for j in range(1, 14):
#             weights_temp[j] += (y - y_Ravenclaw[i]) * x[j][i]
#     weights[0] = weights[0] - learning_rate * weights_temp[0] / m
#     for j in range(1, 14):
#         weights[j] = weights[j] - learning_rate * weights_temp[j] / m

import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def log_loss(y_true, y_pred):
    epsilon = 1e-15  # éviter log(0)
    loss = 0.0
    m = len(y_true)

    for i in range(m):
        y_hat = min(max(y_pred[i], epsilon), 1 - epsilon)
        loss += y_true[i] * math.log(y_hat) + \
                (1 - y_true[i]) * math.log(1 - y_hat)

    return -loss / m

df = pd.read_csv("datasets/dataset_train.csv")

cols_to_drop = ['Index', 'First Name', 'Last Name', 'Birthday']
df.drop(columns=cols_to_drop, inplace=True)

df = pd.get_dummies(df, drop_first=True)
df.fillna(df.mean(), inplace=True)

numeric_cols = df.select_dtypes(include='number').columns
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

y_Ravenclaw = df['Hogwarts House_Ravenclaw'].values
x = df.drop(
    ['Hogwarts House_Ravenclaw',
     'Hogwarts House_Slytherin',
     'Hogwarts House_Hufflepuff'],
    axis=1
)

x = np.hstack((np.ones((x.shape[0], 1)), x.values))
num_features = x.shape[1]
weights = np.zeros(num_features)
learning_rate = 0.01
max_iterations = 10000
m = len(x)

for iteration in range(max_iterations):
    weights_temp = [0.0] * num_features
    predictions = []
    for i in range(m):
        z = sum(a * b for a, b in zip(x[i], weights))
        y_hat = 1 / (1 + math.exp(-z))
        error = y_hat - y_Ravenclaw[i]
        predictions.append(y_hat)
        weights_temp[0] += error
        for j in range(1, num_features):
            weights_temp[j] += error * x[i][j]
    loss = log_loss(y_Ravenclaw, predictions)
    print(f"Iteration {iteration} - Log Loss: {loss}")
    for j in range(num_features):
        weights[j] -= learning_rate * weights_temp[j] / m
    

    



