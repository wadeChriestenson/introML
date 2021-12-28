import numpy
from scipy import stats
import matplotlib.pyplot as plt

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
plt.show()
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
plt.show()
# Histogram Explained
# We use the array from the numpy.random.normal() method, with 100000 values,  to draw a histogram with 100 bars.
# We specify that the mean value is 5.0, and the standard deviation is 1.0.
# Meaning that the values should be concentrated around 5.0, and rarely further away than 1.0 from the mean.
# And as you can see from the histogram, most values are between 4.0 and 6.0, with a top at approximately 5.0.

# Scatter Plot
ageOfCar = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6]
speedOfCar = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]
plt.scatter(ageOfCar, speedOfCar)
plt.show()
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
plt.show()
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
def myfunc(x):
    return slope * x + intercept


# Run each value of the x array through the function. This will result in a new array with new values for the y-axis:
myModel = list(map(myfunc, x))

# Draw the original scatter plot:
plt.scatter(x, y)
# Draw the line of linear regression:
plt.plot(x, myModel)
# Display the diagram:
plt.show()

#  Predict the speed of a car that is 10 years old
speedGuess = myfunc(10)
print(speedGuess)

#  Bad Fit?
# These values for the x- and y-axis should result in a very bad fit for linear regression:
x1 = [89, 43, 36, 36, 95, 10, 66, 34, 38, 20, 26, 29, 48, 64, 6, 5, 36, 66, 72, 40]
y1 = [21, 46, 3, 35, 67, 95, 53, 72, 58, 10, 26, 34, 90, 33, 38, 20, 56, 2, 47, 15]

slope1, intercept1, r1, p1, std_err1 = stats.linregress(x1, y1)


def myfunc1(x1):
    return slope1 * x1 + intercept1

myModel1 = list(map(myfunc1, x1))

plt.scatter(x1, y1)
plt.plot(x1, myModel1)
plt.show()
# You should get a very low r value.
print(r1)
# The result: 0.013 indicates a very bad relationship, and tells us that this data set is not suitable for linear regression.