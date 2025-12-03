# Import readr library to read data
# Import ggplot2 library to plot data for analysis
library(readr)
library(ggplot2)
# For neural network
library(neuralnet)
# For train test split
library(caret)

heart_data <- read.csv("./heart.csv")
heart_data$target <- as.factor(heart_data$target)
levels(heart_data$target) <- c("Normal", "Heart Disease")

View(heart_data)

# Research Question #2: How can resting blood pressure be predicted by other health data?
# Cholesterol, Age, Fasting Blood Sugar > 120mg/dL, ST Depression after exercise

ggplot(heart_data, aes(x=trestbps, y=chol)) + geom_point(color="blue", size=3) +
  geom_smooth(method="lm", se=TRUE, color="red", linetype="solid") + 
  labs(title="Cholesterol Levels by Resting Blood Pressure", x="Resting Blood Pressure", y="Cholesterol (mg/dL)")
cat("Correlation Score Resting Blood Pressure & Cholesterol:", cor(heart_data$trestbps, heart_data$chol, use="complete.obs", method="pearson"))

ggplot(heart_data, aes(x=age, y=trestbps)) + geom_point(color="blue", size=3) +
  geom_smooth(method="lm", se=TRUE, color="red", linetype="solid") + 
  labs(title="Resting Blood Pressure by Age", x="Age", y="Resting Blood Pressure (mm Hg)")
cat("Correlation Score Age & Resting Blood Pressure:", cor(heart_data$age, heart_data$trestbps, use="complete.obs", method="pearson"))

ggplot(heart_data, aes(x=oldpeak, y=trestbps)) + geom_point(color="blue", size=3) +
  geom_smooth(method="lm", se=TRUE, color="red", linetype="solid") + 
  labs(title="Resting Blood Pressure by ST Depression after exercise", x="ST Depression after Exercise", y="Resting Blood Pressure (mm Hg)")
cat("Correlation Score ST Depression & Resting Blood Pressure:", cor(heart_data$oldpeak, heart_data$trestbps, use="complete.obs", method="pearson"))

# To answer research question 2, I will be using a neural network as the plots show nonlinearity
# Scale the data to even out input weight
heart_data_scaled <- scale(heart_data)
heart_data_scaled <- as.data.frame(heart_data_scaled)

# Train test split -- I will be doing 80/20 since there is only around 1000 complete rows of data
train_index <- createDataPartition(heart_data$trestbps, p=.8, list=FALSE)

train_data <- heart_data_scaled[train_index, ]
test_data <- heart_data_scaled[-train_index, ]

# Use cholesterol level, age, and ST depression to predict resting blood pressure
formula <- trestbps ~ chol + age + oldpeak

# Use our formula, training data, hidden layer of two neurons, and a linear output as it's a regression model
nn <- neuralnet(formula, data = train_data, hidden=2, linear.output=TRUE)

#plot(nn)

# Make our first predictions using the neural network
predictions <- compute(nn, test_data[, c("chol", "age", "oldpeak")])

y_test <- test_data$trestbps
y_pred <- predictions$net.result

# Correlation score to measure results
cor(y_pred, y_test)
# Root mean squared error to measure margin of error
mean((y_test-y_pred)^2)^(1/2)

#Prepare resultant data for visualization
result_data <- data.frame(actual=y_test, predicted=y_pred, x=test_data$chol)

#Create a graph showing results
ggplot(result_data) + geom_point(aes(x=x,y=actual,color="Actual"), shape="square") +
  geom_point(aes(x=x,y=predicted, color="Predicted")) +
  labs(x="Cholesterol Level (mg/dL)", y="Resting Blood Pressure (mm Hg)") +
  scale_color_manual(
    name="",
    values=c("Actual"="blue", "Predicted"="red")
  )

# The initial model performed very poorly, so I will be creating a second model including more neurons
# Use our formula, training data, 2 hidden layers of 3 and 2, and a linear output as it's a regression model
nn <- neuralnet(formula, data = train_data, hidden=c(3,2), linear.output=TRUE)

#plot(nn)

# Make our first predictions using the neural network
predictions <- compute(nn, test_data[, c("chol", "age", "oldpeak")])

y_test <- test_data$trestbps
y_pred <- predictions$net.result

# Correlation score to measure results
cor(y_pred, y_test)
# Root mean squared error to measure margin of error
mean((y_test-y_pred)^2)^(1/2)

#Prepare resultant data for visualization
result_data <- data.frame(actual=y_test, predicted=y_pred, x=test_data$chol)

#Create a graph showing results
ggplot(result_data) + geom_point(aes(x=x,y=actual,color="Actual"), shape="square") +
  geom_point(aes(x=x,y=predicted, color="Predicted")) +
  labs(x="Cholesterol Level (mg/dL)", y="Resting Blood Pressure (mm Hg)") +
  scale_color_manual(
    name="",
    values=c("Actual"="blue", "Predicted"="red")
  )

# Create a density plot to compare distribution of predicted data to true data
ggplot(result_data) + geom_density(aes(x=y_test, color="Actual")) +
  geom_density(aes(x=y_pred, color="Predicted")) +
  labs(title="Distribution of Predicted & True Resting Blood Pressure",
       x="Resting Blood Pressure (mm Hg)",
       y="Density"
       ) + 
  scale_color_manual(
    name="",
    values=c("Actual"="blue", "Predicted"="red")
  )
# The density plot shows that the model succeeds to predict resting blood pressure to be close to the median,
# as the peaks are similar, but the model struggles to predict those that land outside of the inner-quartile
# range

### Research Question #2 Answer
# Other data from the dataset can be used to reasonably "Ballpark" resting blood pressure
# The data can not precisely predict resting blood pressure using the neural network created
# The model cannot converge with beyond 3 hidden layers or more than 5 neurons
# Discussion idea: If a higher budget for convergence were used, how much more accurate could the model be?

heart_data_female <- heart_data[heart_data$sex==0, ]
heart_data_male <- heart_data[heart_data$sex==1, ]
View(heart_data_female)
View(heart_data_male)

# Plot the distribution of Heartrate data by normal or heart disease

ggplot(heart_data_female) + geom_boxplot(aes(x=thalach, y=target, fill=target)) + scale_x_continuous(limits=c(80,220)) +
  labs(title="Female Distribution of Maximum Recorded Heartrate", x="Maximum Recorded Heartrate (BPM)",
       y="Heart Disease Status")
ggplot(heart_data_male) + geom_boxplot(aes(x=thalach, y=target, fill=target)) + scale_x_continuous(limits=c(80,220)) +
  labs(title="Male Distribution of Maximum Recorded Heartrate", x="Maximum Recorded Heartrate (BPM)",
       y="Heart Disease Status")

# Plot the distribution of ST Depression After Exercise

ggplot(heart_data_female) + geom_boxplot(aes(x=oldpeak, y=target, fill=target)) + scale_x_continuous(limits=c(0,6.5)) +
  labs(title="Female Distribution of ST Depression After Exercise", x="ST Depression After Exercise (mm)",
       y="Heart Disease Status")
ggplot(heart_data_male) + geom_boxplot(aes(x=oldpeak, y=target, fill=target)) + scale_x_continuous(limits=c(0,6.5)) +
  labs(title="Male Distribution of ST Depression After Exercise", x="ST Depression After Exercise (mm)",
       y="Heart Disease Status")

# Plot the distribution of Reported Chest Pain

ggplot(heart_data_female) + geom_boxplot(aes(x=cp, y=target, fill=target)) + scale_x_continuous(limits=c(-.5,3.5)) +
  labs(title="Female Distribution of Reported Chest Pain", x="Reported Chest Pain (Scale of 0-3)",
       y="Heart Disease Status")
ggplot(heart_data_male) + geom_boxplot(aes(x=cp, y=target, fill=target)) + scale_x_continuous(limits=c(-.5,3.5)) +
  labs(title="Male Distribution of Reported Chest Pain", x="Reported Chest Pain (Scale of 0-3)",
       y="Heart Disease Status")

# Create clustered histogram to show distribution of reported chest pain

ggplot(heart_data_female, aes(x=target, fill=factor(cp))) + geom_bar(aes(y=after_stat((count/sum(count))*100)), 
  position="fill") + labs(fill="Reported Chest Pain (0-3)", y="Percentage", x="Heart Disease Status", 
    title="Female Distribution of Reported Chest Pain by Heart Disease Status") + scale_y_continuous(limits=c(0,50))
ggplot(heart_data_male, aes(x=target, fill=factor(cp))) + geom_bar(aes(y=after_stat((count/sum(count))*100)), 
  position="fill") + labs(fill="Reported Chest Pain (0-3)", y="Percentage", x="Heart Disease Status", 
    title="Male Distribution of Reported Chest Pain by Heart Disease Status") + scale_y_continuous(limits=c(0,50))