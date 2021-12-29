import numpy
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.image as pltimg

speed = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86, 45, 56, 72, 53, 68, 75, 46, 59, 120, 81, 73]
print('Speed Array: ', speed)
# Numpy method to find the mean/avg of the array
mean = numpy.mean(speed)
print('Mean:', mean)

# Numpy method to find the median/mid-point of the array
median = numpy.median(speed)
print('Median: ', median)

# SciPi method to find the mode/most-common-value the array
mode = stats.mode(speed)
print('Mode: ', mode)

# Numpy method to find the standard deviation the array
# Standard Deviation is often represented by the symbol Sigma: σ
# Multiply the standard deviation by itself, you get the variance!
stdDevi = numpy.std(speed)
print('Standard Deviation: ', stdDevi)

# Numpy method to find the variance of the array
# Variance is often represented by the symbol Sigma Square: σ2
# The square root of the variance, is the standard deviation!
variance = numpy.var(speed)
print('Variance: ', variance)

# Numpy method to find the percentile of the array,
# also need the percent you are looking for as the second argument
percentile = numpy.percentile(speed, 75)
print('Percentile: ', percentile)

# Numpy method to create random datasets.
# Create an array containing 250 random floats between 0 and 5:
dataSet0 = numpy.random.uniform(0.0, 5, 1000)
print(dataSet0)

# Data Distribution

# Matlotlib method to draw a histogram of dataSet0
plt.hist(dataSet0, 5)
# plt.show()
# Histogram Explained
# We use the array from the example above to draw a histogram with 5 bars.
# The first bar represents how many values in the array are between 0 and 1.
# The second bar represents how many values are between 1 and 2.
# Etc.
# Which gives us this result:
# 52 values are between 0 and 1
# 48 values are between 1 and 2
# 49 values are between 2 and 3
# 51 values are between 3 and 4
# 50 values are between 4 and 5

#  Normal Data Distribution
# We will now learn how to create an array where the values are concentrated on a given value.
# In probability theory this kind of data distribution is known as the normal data distribution,
# or the Gaussian data distribution, after the mathematician Carl Friedrich Gauss who came up with the formula of this data distribution.

# Numpy method to create a normal data distribution
dataSetNormal = numpy.random.normal(5.0, 1.0, 100000)
plt.hist(dataSetNormal, 100)
# plt.show()
# Histogram Explained
# We use the array from the numpy.random.normal() method, with 100000 values,  to draw a histogram with 100 bars.
# We specify that the mean value is 5.0, and the standard deviation is 1.0.
# Meaning that the values should be concentrated around 5.0, and rarely further away than 1.0 from the mean.
# And as you can see from the histogram, most values are between 4.0 and 6.0, with a top at approximately 5.0.

# Scatter Plot
ageOfCar = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6]
speedOfCar = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]
plt.scatter(ageOfCar, speedOfCar)
# plt.show()
# Scatter Plot Explained
# The x-axis represents ages, and the y-axis represents speeds.
# What we can read from the diagram is that the two fastest cars were both 2 years old, and the slowest car was 12 years old.


# Random Data Distribution
# Create two arrays that are both filled with 1000 random numbers from a normal data distribution.
# The first array will have the mean set to 5.0 with a standard deviation of 1.0.
# The second array will have the mean set to 10.0 with a standard deviation of 2.0:
x1 = numpy.random.normal(5.0, 1.0, 1000)
y1 = numpy.random.normal(10.0, 2.0, 1000)
plt.scatter(x1, y1)
# plt.show()
# Scatter Plot Explained
# We can see that the dots are concentrated around the value 5 on the x-axis, and 10 on the y-axis.
# We can also see that the spread is wider on the y-axis than on the x-axis.

# Machine Learning - Linear Regression
# The term regression is used when you try to find the relationship between variables.
#
# In Machine Learning, and in statistical modeling, that relationship is used to predict the outcome of future events.
# Linear Regression uses the relationship between the data-points to draw a straight line through all them.
# This line can be used to predict future values.
# How Does it Work?
# Python has methods for finding a relationship between data-points and to draw a line of linear regression.
# We will show you how to use these methods instead of going through the mathematic formula.
# In the example below, the x-axis represents age, and the y-axis represents speed.
# We have registered the age and speed of 13 cars as they were passing a tollbooth.
# Let us see if the data we collected could be used in a linear regression:

# Create the arrays that represent the values of the x and y axis:
x = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6]
y = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]

# Execute a method that returns some important key values of Linear Regression:
slope, intercept, r, p, std_err = stats.linregress(x, y)


# R for Relationship
# It is important to know how the relationship between the values of the x-axis and the values of the y-axis is,
# if there are no relationship the linear regression can not be used to predict anything.
# This relationship - the coefficient of correlation - is called r.
# The r value ranges from -1 to 1, where 0 means no relationship, and 1 (and -1) means 100% related.
# Python and the Scipy module will compute this value for you, all you have to do is feed it with the x and y values.


# Create a function that uses the slope and intercept values to return a new value.
# This new value represents where on the y-axis the corresponding x value will be placed:
def Func(x):
    return slope * x + intercept


# Run each value of the x array through the function. This will result in a new array with new values for the y-axis:
myModel = list(map(Func, x))

# Draw the original scatter plot:
plt.scatter(x, y)
# Draw the line of linear regression:
plt.plot(x, myModel)
# Display the diagram:
# plt.show()

#  Predict the speed of a car that is 10 years old
speedGuess = Func(10)
print(speedGuess)

#  Bad Fit?
# These values for the x- and y-axis should result in a very bad fit for linear regression:
x1 = [89, 43, 36, 36, 95, 10, 66, 34, 38, 20, 26, 29, 48, 64, 6, 5, 36, 66, 72, 40]
y1 = [21, 46, 3, 35, 67, 95, 53, 72, 58, 10, 26, 34, 90, 33, 38, 20, 56, 2, 47, 15]

slope1, intercept1, r1, p1, std_err1 = stats.linregress(x1, y1)


def Func1(x1):
    return slope1 * x1 + intercept1


myModel1 = list(map(Func1, x1))

plt.scatter(x1, y1)
plt.plot(x1, myModel1)
# plt.show()
# You should get a very low r value.
print(r1)
# The result: 0.013 indicates a very bad relationship, and tells us that this data set is not suitable for linear regression.

# Machine Learning - Polynomial Regression
# Polynomial Regression
# If your data points clearly will not fit a linear regression (a straight line through all data points),
# it might be ideal for polynomial regression.
# Polynomial regression, like linear regression,
# uses the relationship between the variables x and y to find the best way to draw a line through the data points.

# How Does it Work?
# Python has methods for finding a relationship between data-points and to draw a line of polynomial regression.
# We will show you how to use these methods instead of going through the mathematical formula.
# In the example below, we have registered 18 cars as they were passing a certain tollbooth.
# We have registered the car's speed, and the time of day (hour) the passing occurred.
# The x-axis represents the hours of the day and the y-axis represents the speed:
x2 = [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 21, 22]
y2 = [100, 90, 80, 60, 60, 55, 60, 65, 70, 70, 75, 76, 78, 79, 90, 99, 99, 100]
# NumPy has a method that lets us make a polynomial model:
myModel2 = numpy.poly1d(numpy.polyfit(x2, y2, 3))
# Then specify how the line will display, we start at position 1, and end at position 22:
myLine2 = numpy.linspace(1, 22, 100)

plt.scatter(x2, y2)
plt.plot(myLine2, myModel2(myLine2))
plt.scatter(x2, y2)
# plt.show()
# R-Squared
# It is important to know how well the relationship between the values of the x- and y-axis is,
# if there are no relationship the polynomial regression can not be used to predict anything.
# The relationship is measured with a value called the r-squared.
# The r-squared value ranges from 0 to 1, where 0 means no relationship, and 1 means 100% related.
# Python and the Sklearn module will compute this value for you, all you have to do is feed it with the x and y arrays:
myModel3 = numpy.poly1d(numpy.polyfit(x2, y2, 3))

print(r2_score(y2, myModel3(x2)))
# Note: The result 0.94 shows that there is a very good relationship, and we can use polynomial regression in future predictions.
# Predict Future Values
# Now we can use the information we have gathered to predict future values.
# Example: Let us try to predict the speed of a car that passes the tollbooth at around 17 P.M:
# To do so, we need the same myModel2 array from the example above:
myModel4 = numpy.poly1d(numpy.polyfit(x2, y2, 3))
speed2 = myModel4(17)
print(speed2)

# Multiple Regression
# Multiple regression is like linear regression, but with more than one independent value, meaning that we try to predict a value based on two or more variables.

df = pd.read_csv("cars.csv")
# Then make a list of the independent values and call this variable X.
# Put the dependent values in a variable called y.
X = df[['Weight', 'Volume']]
y = df['CO2']
# From the sklearn module we will use the LinearRegression() method to create a linear regression object.
# This object has a method called fit() that takes the independent and dependent values as parameters and fills the regression object with data that describes the relationship:
regr = linear_model.LinearRegression()
regr.fit(X, y)

# predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm3:
predictedCO2 = regr.predict([[2300, 1300]])

print(predictedCO2)
# Coefficient
# The coefficient is a factor that describes the relationship with an unknown variable.
# Example: if x is a variable, then 2x is x two times. x is the unknown variable, and the number 2 is the coefficient.
# In this case, we can ask for the coefficient value of weight against CO2, and for volume against CO2.
# The answer(s) we get tells us what would happen if we increase, or decrease, one of the independent values.
print(regr.coef_)
# Result Explained
# The result array represents the coefficient values of weight and volume.
# Weight: 0.00755095
# Volume: 0.00780526
# These values tell us that if the weight increase by 1kg, the CO2 emission increases by 0.00755095g.
# And if the engine size (Volume) increases by 1 cm3, the CO2 emission increases by 0.00780526 g.
# I think that is a fair guess, but let test it!
# We have already predicted that if a car with a 1300cm3 engine weighs 2300kg, the CO2 emission will be approximately 107g.


# Scale Features
# When your data has different values, and even different measurement units, it can be difficult to compare them. What is kilograms compared to meters? Or altitude compared to time?
# The answer to this problem is scaling. We can scale data into new values that are easier to compare.
# Take a look at the table below, it is the same data set that we used in the multiple regression chapter,
# but this time the volume column contains values in liters instead of cm3 (1.0 instead of 1000).
# It can be difficult to compare the volume 1.0 with the weight 790, but if we scale them both into comparable values, we can easily see how much one value is compared to the other.
# There are different methods for scaling data, in this tutorial we will use a method called standardization.
# The standardization method uses this formula:
# z = (x - u) / s
# Where z is the new value, x is the original value, u is the mean and s is the standard deviation.
# If you take the weight column from the data set above, the first value is 790, and the scaled value will be:
# (790 - 1292.23) / 238.74 = -2.1
# If you take the volume column from the data set above, the first value is 1.0, and the scaled value will be:
# (1.0 - 1.61) / 0.38 = -1.59
# Now you can compare -2.1 with -1.59 instead of comparing 790 with 1.0.
scale = StandardScaler()
df = pd.read_csv("cars2.csv")
X1 = df[['Weight', 'Volume']]
scaledX = scale.fit_transform(X)

print(scaledX)
# Predict CO2 Values
# When the data set is scaled, you will have to use the scale when you predict values:
scale = StandardScaler()

df = pd.read_csv("cars2.csv")

X = df[['Weight', 'Volume']]
y = df['CO2']

scaledX = scale.fit_transform(X)

regr = linear_model.LinearRegression()
regr.fit(scaledX, y)

scaled = scale.transform([[2300, 1.3]])

predictedCO2 = regr.predict([scaled[0]])
print(predictedCO2)

Machine Learning - Train/Test
What is Train/Test
Train/Test is a method to measure the accuracy of your model.
It is called Train/Test because you split the the data set into two sets: a training set and a testing set.
80% for training, and 20% for testing.
You train the model using the training set.
You test the model using the testing set.
Train the model means create the model.
Test the model means test the accuracy of the model.
Start With a Data Set
Start with a data set you want to test.
Our data set illustrates 100 customers in a shop, and their shopping habits.
numpy.random.seed(2)

x = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100) / x

plt.scatter(x, y)
# plt.show()

# Split Into Train/Test
# The training set should be a random selection of 80% of the original data.
# The testing set should be the remaining 20%.
train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

# Display the Training Set
# Display the same scatter plot with the training set:

plt.scatter(train_x, train_y)
# plt.show()

# Display the Testing Set
# To make sure the testing set is not completely different,
# we will take a look at the testing set as well.

plt.scatter(test_x, test_y)
# plt.show()

# Fit the Data Set
# What does the data set look like? In my opinion I think the best fit would be a polynomial regression,
# so let us draw a line of polynomial regression.
# To draw a line through the data points, we use the plot() method of the matplotlib module:
mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))

myline = numpy.linspace(0, 6, 100)

plt.scatter(train_x, train_y)
plt.plot(myline, mymodel(myline))
plt.show()

# The result can back my suggestion of the data set fitting a polynomial regression,
# even though it would give us some weird results if we try to predict values outside of the data set.
# Example: the line indicates that a customer spending 6 minutes in the shop would make a purchase worth 200.
# That is probably a sign of overfitting.
# But what about the R-squared score? The R-squared score is a good indicator of how well my data set is fitting the model.
# R2
# Remember R2, also known as R-squared?
# It measures the relationship between the x axis and the y axis, and the value ranges from 0 to 1,
# where 0 means no relationship, and 1 means totally related.
# The sklearn module has a method called r2_score() that will help us find this relationship.
# In this case we would like to measure the relationship between the minutes a customer,
# stays in the shop and how much money they spend.
mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))

r2 = r2_score(train_y, mymodel(train_x))

print(r2)
# Note: The result 0.799 shows that there is a OK relationship.

# Bring in the Testing Set
# Now we have made a model that is OK, at least when it comes to training data.
# Now we want to test the model with the testing data as well, to see if gives us the same result.

mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))

r2 = r2_score(test_y, mymodel(test_x))

print(r2)
# Note: The result 0.809 shows that the model fits the testing set as well,
# and we are confident that we can use the model to predict future values.

# Predict Values
# Now that we have established that our model is OK, we can start predicting new values.
# How much money will a buying customer spend, if she or he stays in the shop for 5 minutes?
print(mymodel(5))

# Machine Learning - Decision Tree
# Decision Tree
# In this chapter we will show you how to make a "Decision Tree".
# A Decision Tree is a Flow Chart, and can help you make decisions based on previous experience.
# In the example, a person will try to decide if he/she should go to a comedy show or not.
# Luckily our example person has registered every time there was a comedy show in town,
# and registered some information about the comedian, and also registered if he/she went or not.
# How Does it Work?

df = pd.read_csv("shows.csv")

print(df)

# To make a decision tree, all data has to be numerical.
# We have to convert the non numerical columns 'Nationality' and 'Go' into numerical values.
# Pandas has a map() method that takes a dictionary with information on how to convert the values.
# {'UK': 0, 'USA': 1, 'N': 2}
# Means convert the values 'UK' to 0, 'USA' to 1, and 'N' to 2.

d = {'UK': 0, 'USA': 1, 'N': 2}
df['Nationality'] = df['Nationality'].map(d)
d = {'YES': 1, 'NO': 0}
df['Go'] = df['Go'].map(d)

print(df)

# Then we have to separate the feature columns from the target column.
# The feature columns are the columns that we try to predict from,
# and the target column is the column with the values we try to predict.

features = ['Age', 'Experience', 'Rank', 'Nationality']

X = df[features]
y = df['Go']

print(X)
print(y)

# Now we can create the actual decision tree, fit it with our details, and save a .png file on the computer:


dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)
data = tree.export_graphviz(dtree, out_file=None, feature_names=features)
graph = pydotplus.graph_from_dot_data(data)
graph.write_png('mydecisiontree.png')

img = pltimg.imread('mydecisiontree.png')
imgplot = plt.imshow(img)
plt.show()
