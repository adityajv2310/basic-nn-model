### EXP NO: 01
### DATE: 29.08.2022

# <p align = "center"> Developing a Neural Network Regression Model

## AIM:

To develop a neural network regression model for the given dataset.

## THEORY:

The Neural network model contains input layer,two hidden layers and output layer.Input layer contains a single neuron.Output layer also contains single neuron.First hidden layer contains six neurons and second hidden layer contains seven neurons.A neuron in input layer is connected with every neurons in a first hidden layer.Similarly,each neurons in first hidden layer is connected with all neurons in second hidden layer.All neurons in second hidden layer is connected with output layered neuron.Relu activation function is used here .It is linear neural network model(single input neuron forms single output neuron).

## Neural Network Model:

![Dl Exp1](https://user-images.githubusercontent.com/75235386/187220195-e22c99aa-5ba2-42c7-83d4-5da0d125fc7a.jpeg)

## DESIGN STEPS:

### STEP 1:

Loading the dataset.

### STEP 2:

Split the dataset into training and testing.

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot.

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM:
```
Developed By: Aditya JV
Register Number: 212220230002
```
```Python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
data=pd.read_csv("dataset.csv")
data.head()
x=data[['Input']].values
x
y=data[['Output']].values
y
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=33)
Scaler=MinMaxScaler()
Scaler.fit(x_train)
Scaler.fit(x_test)
x_train1=Scaler.transform(x_train)
x_train1
x_train
AI_BRAIN=Sequential([
    Dense(6,activation='relu'),
    Dense(7,activation='relu'),
    Dense(1,activation='relu')
])
AI_BRAIN.compile(optimizer='rmsprop', loss='mse')
AI_BRAIN.fit(x_train1,y_train,epochs=2000)
loss_df=pd.DataFrame(AI_BRAIN.history.history)
loss_df.plot()
x_test1=Scaler.transform(x_test)
x_test1
AI_BRAIN.evaluate(x_test1,y_test)
x_n1=[[50]]
x_n1_1=Scaler.transform(x_n1)
AI_BRAIN.predict(x_n1_1)

```

## Dataset Information:

![Dataset1](https://user-images.githubusercontent.com/75235386/187226492-880a85d1-c4a8-47c9-898a-180837152423.jpg)

## OUTPUT:

### Training Loss Vs Iteration Plot

![Loss vs Iteration Plot](https://user-images.githubusercontent.com/75235386/187227643-9825decb-6d9a-41b8-9bb9-218f9f4c0db6.jpg)

### Test Data Root Mean Squared Error

![Test data rmse](https://user-images.githubusercontent.com/75235386/187229952-4f15d071-40df-462f-b9c4-038847e6a216.jpg)

### New Sample Data Prediction

![New sample data](https://user-images.githubusercontent.com/75235386/187230023-1ecb1ecb-923e-48b2-9b58-32ad392d4824.jpg)

## RESULT:
Thus, the neural network model for the given dataset is implemented successfully.
