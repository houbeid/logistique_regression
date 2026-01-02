import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("datasets/dataset_train.csv")

houses = ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"]

features = ["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts", "Divination", "Muggle Studies", "Ancient Runes", "History of Magic", "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying"]

for feature in features:
    for house in houses:
        scores = df[df["Hogwarts House"] == house][feature].dropna()
        plt.hist(scores, bins=20, alpha=0.5, label=house)
    
    plt.title(f"{feature} - Distribution des scores par maison")
    plt.xlabel("Score")
    plt.ylabel("Nombre d'élèves")
    plt.legend()
    plt.show()