# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: NITHISH S
RegisterNumber:  212223220070
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:

df.head()

![dfhead]![229978451-2b6bdc4f-522e-473e-ae2f-84ec824344c5](https://github.com/Nithish23013509/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149038138/3927101b-bd2f-4d38-85f9-3658e53db72f)


df.tail()

![dftail]![229978854-6af7d9e9-537f-4820-a10b-ab537f3d0683](https://github.com/Nithish23013509/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149038138/eb37ace8-cde2-4293-a4ab-758912b03cb8)


Array value of X

![xvalue]![229978918-707c006d-0a30-4833-bf77-edd37e8849bb](https://github.com/Nithish23013509/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149038138/521efad4-5249-4fb0-9582-a9140a30fe6f)


Array value of Y

![yvalue]![229978994-b0d2c87c-bef9-4efe-bba2-0bc57d292d20](https://github.com/Nithish23013509/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149038138/a49217f8-f5d1-4426-b708-5d6b6fcd2a96)


Values of Y prediction

![ypred]![229979053-f32194cb-7ed4-4326-8a39-fe8186079b63](https://github.com/Nithish23013509/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149038138/a33af6c1-3ab8-40a3-9edb-2a0f804e9971)


Array values of Y test

![ytest]![229979114-3667c4b7-7610-4175-9532-5538b83957ac](https://github.com/Nithish23013509/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149038138/476e3976-6965-4fb3-abe1-6ea2228d9ba6)


Training Set Graph

![train])![229979169-ad4db5b6-e238-4d80-ae5b-405638820d35](https://github.com/Nithish23013509/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149038138/f37fb65c-d76d-40eb-ab98-2262530ae74d)


Test Set Graph

![test]![229979225-ba90853c-7fe0-4fb2-8454-a6a0b921bdc1](https://github.com/Nithish23013509/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149038138/33e3806a-1159-403c-a89a-93f0e5b40e49)


Values of MSE, MAE and RMSE

![mse]![229979276-bb9ffc68-25f8-42fe-9f2a-d7187753aa1c](https://github.com/Nithish23013509/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149038138/6f6f5e05-1001-4552-8cb6-11cb00f5a824)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
