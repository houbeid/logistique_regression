import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("datasets/dataset_train.csv")
features = ["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts", "Divination", "Muggle Studies", "Ancient Runes", "History of Magic", "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying"]
for i in range(len(features)):
    for j  in range(i + 1, len(features)):
        plt.scatter(df[features[i]], df[features[j]])
        plt.xlabel(features[i])
        plt.ylabel(features[j])
        plt.title(f"Relation entre {features[i]} et {features[j]}")
        plt.show()