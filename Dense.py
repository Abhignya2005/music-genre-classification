import pandas as pd
from sklearn.neural_network import MLPClassifier  
import train_test_split

# Define an MLP with dense layers
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', max_iter=500)

data = pd.DataFrame({
    'Feature1': [1,2,3,4,5,6,7,8,9,10],
    'Feature2': [10,20,30,40,50,60,70,80,90,100],
    'Label': [0,1,0,1,0,1,0,1,1,1]
})


X = data [['Feature1', 'Feature2']] #Feature
Y = data['Label'] # label 

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Example training (X is input data, y are labels)
mlp.fit(X_train, y_train)

# Predictions
predictions = mlp.predict(X_test)
