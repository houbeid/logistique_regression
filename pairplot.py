import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("datasets/dataset_train.csv")
df.drop(columns=['Index', 'First Name', 'Last Name', 'Birthday'], inplace=True)
sns.pairplot(df, hue='Hogwarts House', diag_kind='kde')
plt.show()