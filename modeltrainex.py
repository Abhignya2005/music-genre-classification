import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.DataFrame({
    'Feature1': [1,2,3,4,5,6,7,8,9,10],
    'Feature2': [10,20,30,40,50,60,70,80,90,100],
    'Label': [0,1,0,1,0,1,0,1,1,1]
})

X = data [['Feature1', 'Feature2']] #Feature
Y = data['Label'] # label 

#spilt into training and test sets [80 % training and 20% testing]
# X_train , X_test , y_train , y_test = train_test_split(X, y, test_size =0.2, random_state = 10 )

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

print("Training Fetaure:\n",X_train)
print("\nTesting Feature:\n",X_test)
print("\nTraining Lables:\n",y_train)
print("\nTesting Labels:\n",y_test)



"""X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10) output for this 
#Training Fetaure:
#   Feature1  Feature2
# 5         6        60
# 6         7        70
# 3         4        40
# 1         2        20
# 0         1        10
# 7         8        80
# 4         5        50
# 9        10       100

# Testing Feature:
#    Feature1  Feature2
# 8         9        90
# 2         3        30

# Training Lables:
#  5    1
# 6    0
# 3    1
# 1    1
# 0    0
# 7    1
# 4    0
# 9    1
# Name: Label, dtype: int64
# 
# Testing Labels:
# 8    1
# 2    0
# Name: Label, dtype: int64 """

"""X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) output for this 
#Training Fetaure:
#    Feature1  Feature2
#2         3        30
#5         6        60
#1         2        20
#6         7        70
#4         5        50
#7         8        80
#9        10       100
#8         9        90

#Testing Feature:
#    Feature1  Feature2
#3         4        40
#0         1        10

#Training Lables:
# 2    0
#5    1
#1    1
#6    0
#4    0
#7    1
#9    1
#8    1
#Name: Label, dtype: int64

#Testing Labels:
#3    1
#0    0
#Name: Label, dtype: int64"""