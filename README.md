# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
#NAME:DHANUSH RAJAN .T
#REF.no:250137437
## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```



Code
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
Matplotlib is building the font cache; this may take a moment.
data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks_Scored":  [35, 40, 50, 55, 60, 65, 70, 80, 85, 95]
}
df = pd.DataFrame(data)
​
# Display dataset
print("Dataset:\n", df.head())
df
Dataset:
    Hours_Studied  Marks_Scored
0              1            35
1              2            40
2              3            50
3              4            55
4              5            60
Hours_Studied	Marks_Scored
0	1	35
1	2	40
2	3	50
3	4	55
4	5	60
5	6	65
6	7	70
7	8	80
8	9	85
9	10	95
X = df[["Hours_Studied"]]   # Independent variable
y = df["Marks_Scored"]      # Dependent variable
​
# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)

LinearRegression
LinearRegression()
y_pred = model.predict(X_test)
​
# Step 7: Model Evaluation
print("\nModel Parameters:")
print("Intercept (b0):", model.intercept_)
print("Slope (b1):", model.coef_[0])
​
print("\nEvaluation Metrics:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

Model Parameters:
Intercept (b0): 28.663793103448278
Slope (b1): 6.379310344827586

Evaluation Metrics:
Mean Squared Error: 1.5922265160523277
R² Score: 0.9968548612028596
plt.figure(figsize=(8,6))
plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, model.predict(X), color='red', linewidth=2, label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Simple Linear Regression: Predicting Marks")
plt.legend()
plt.grid(True)
plt.show()
​

hours = 7.5
predicted_marks = model.predict([[hours]])
print(f"\nPredicted marks for {hours} hours of study = {predicted_marks[0]:.2f}")
​

Predicted marks for 7.5 hours of study = 76.51
C:\ProgramData\anaconda3\lib\site-packages\sklearn\base.py:420: UserWarning: X does not have valid feature names, but LinearRegression was fitted with feature names
  warnings.warn(


```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
<img width="902" height="685" alt="image" src="https://github.com/user-attachments/assets/dd3bdee0-774c-4084-bbae-9bf475f38322" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
