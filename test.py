import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Create the dataset
data = {
    "Math": [45, 75, 60, 90, 35, 70],
    "English": [55, 80, 50, 85, 45, 60],
    "Science": [40, 85, 55, 80, 40, 75],
    "History": [50, 70, 45, 88, 30, 65],
    "Pass_Fail": [0, 1, 0, 1, 0, 1]  # 0 = Fail, 1 = Pass
}
df = pd.DataFrame(data)

# Features and target
X = df[["Math", "English", "Science", "History"]]
y = df["Pass_Fail"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# New student marks
new_student = pd.DataFrame({
    "Math": [50],
    "English": [60],
    "Science": [55],
    "History": [40]
})

# Prediction
prediction = model.predict(new_student)
result = "Pass" if prediction[0] == 1 else "Fail"

print(f"The predicted result for the new student is: {result}")
