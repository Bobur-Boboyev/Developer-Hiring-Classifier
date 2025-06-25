from sklearn.linear_model import LogisticRegression
import numpy as np

data = np.loadtxt("data/developer.csv", delimiter=",", skiprows=1)

x = data[:, :-1]
y = data[:, -1]

model = LogisticRegression()
model.fit(x, y)

try:
    age = int(input("Age: "))
    ielts = float(input("IELTS score: "))
    projects = int(input("Completed projects: "))
    soft = int(input("Soft skill score (1-10): "))
    hard = int(input("Hard skill score (1-10): "))
    

    features = np.array([[age,ielts,projects,soft,hard]])
    prediction = model.predict(features)[0]

    print("\nPrediction: Hired" if prediction == 1 else "\nPrediction: Not Hired")

except Exception as e:
    print("An error occurred:", e)


