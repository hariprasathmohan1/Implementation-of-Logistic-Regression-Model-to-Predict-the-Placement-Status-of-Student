# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the placement dataset and remove unnecessary columns such as serial number and salary.
2. Convert all categorical attributes into numerical values using label encoding.
3. Split the dataset into training and testing sets using train–test split.
4.Train a Logistic Regression model using the training data.
5..Predict placement status and evaluate the model using accuracy and confusion matrix.

## Program:
```

Developed by: M YOGESWARI
RegisterNumber:  212224040373

import pandas as pd
import numpy as np

df = pd.read_csv('/content/Placement_Data.csv')
df

df1 = df.copy()
df1

df1 = df1.drop(['sl_no', 'salary'], axis=1)

df1.isnull().sum()

df1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df1['gender'] = le.fit_transform(df1['gender'])
df1['ssc_b'] = le.fit_transform(df1['ssc_b'])
df1['hsc_b'] = le.fit_transform(df1['hsc_b'])
df1['hsc_s'] = le.fit_transform(df1['hsc_s'])
df1['degree_t'] = le.fit_transform(df1['degree_t'])
df1['workex'] = le.fit_transform(df1['workex'])
df1['specialisation'] = le.fit_transform(df1['specialisation'])
df1['status'] = le.fit_transform(df1['status'])

df1

x = df1.iloc[:, :-1]
x

y = df1['status']
y

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='liblinear')

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy = accuracy_score(y_test, y_pred)

confusion = confusion_matrix(y_test, y_pred)

cr = classification_report(y_test, y_pred)

print("Accuracy Score:", accuracy)

print("\nConfusion Matrix:\n", confusion)

print("\nClassification Report:\n", cr)

from sklearn import metrics

cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=confusion,
    display_labels=['true', 'false']
)

cm_display.plot()

```

## Output:

<img width="1148" height="846" alt="image" src="https://github.com/user-attachments/assets/d2bb357c-8f89-4963-9f48-5255c1df0187" />

<img width="886" height="710" alt="image" src="https://github.com/user-attachments/assets/fb63185d-5cc1-43f0-9074-f682648f23da" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
