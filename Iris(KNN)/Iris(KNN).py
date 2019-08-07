
#Import the necessary libraries
import pandas as pd

#Storing the location of the dataset in a variable
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

#Assigning column names to the dataset
names = ['sepal-length','sepal-width','petal-length','petal-width','Class']

#Read the dataset using pandas
data = pd.read_csv(url, names =names)

#Store the independent variables of the dataset in a variable
X = data.loc[:,['sepal-length','sepal-width','petal-length','petal-width']].values

#Store the target variable
y = data['Class'].values

#The dataset must now be splitted into testing and training 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

#Normalizing to bring the attributes to a same scale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Importing KNeighboursClassifier to perform KNN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5)

classifier.fit(X_train,y_train)

#Predict the result
predict= classifier.predict(X_test)

#Print the accuracy score
print(classifier.score(X_test,y_test)*100)

#To compare the result, display the predicted value and the expected value 
for i in range(len(predict)):
    print("Predicted value is: ",predict[i]," and Expected value is: ",y_test[i])
    


