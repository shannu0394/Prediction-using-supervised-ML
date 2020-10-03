# Prediction-using-supervised-ML
Prediction of percentage of marks of an student based on number of study hours
Linear Regression with Python Scikit Learn
In this section we will see how the Python Scikit-Learn library for machine learning can be used to implement regression functions.we will start with simple linear regression involving two variables.
Simple Linear Regression
In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. This is a simple linear regression task as it involves just two variables.
#importing all modules required for this project
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
#importing all modules required for this project
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
#Reading data
data = pd.read_csv("student_scores.csv")
print("Data imported successfully")
data.head(10)
Data imported successfully
Hours	Scores
0	2.5	21
1	5.1	47
2	3.2	27
3	8.5	75
4	3.5	30
5	1.5	20
6	9.2	88
7	5.5	60
8	8.3	81
9	2.7	25
#plotting the distribution of scores
data.plot(x="Hours",y="Scores",style="o")
plt.title("Hours vs Percentage")
plt.xlabel("Hours studied")
plt.ylabel("Percentage score")
plt.show()

From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score.
Preparing Data
The next step is to divide the data into "attributes" (inputs) and "labels" (outputs).
X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values
Now that we have our attributes and labels, the next step is to split this data into training and test sets. We'll do this by using scikit-learn's built-in train_test_split() method.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)
Training the algorithm
We have split our data into training and test sets, and now is finally the time to train our alogrithm
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
print("Training complete.")
Training complete.
#Plotting the regression line
line = regressor.coef_*X+regressor.intercept_
​
#Plotting for the test data
plt.scatter(X,y)
plt.plot(X, line)
plt.show()

Making Predictions
Now that we have trained our algorithm, it's time to make some predictions.
print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores
[[1.5]
 [3.2]
 [7.4]
 [2.5]
 [5.9]]
# Comparing actual vs predicted
df = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
df
Actual	Predicted
0	20	16.884145
1	27	33.732261
2	69	75.357018
3	30	26.794801
4	62	60.491033
# Testing with own data
hours = 9.52
test = np.array([hours])
test = test.reshape(-1,1)
pred = regressor.predict(test)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(pred[0]))
No of Hours = 9.52
Predicted Score = 96.3676097371488
# if the student studies for 9.52 hours per day  then the predicted percentage is 96.36
​
Evaluating the model
The final step is to evaluate the performance of algorithm. This step is particularly important to compare how well different algorithms perform on a particular dataset. For simplicity here, we have chosen the mean square error. There are many such metrics.
from sklearn import metrics
print("Mean absolute error:", metrics.mean_absolute_error(y_test,y_pred))
​
Mean absolute error: 4.183859899002982
