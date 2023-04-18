# Wine Project: Clustering

## Description
- This project will use clustering methods in combination with supervised machine learning algorithms to predict the quality of wine using a dataset acquired from data.world. It will go through the data science pipeline and create a model that accurately predicts the quality of wine.

## Goals
- Predict the quality of wine using features in the dataset
  - Construct a ML model that accurately predicts wine quality
  - Deliver a technical report that a data scientist can read through and understand 

## Initial Questions
What is the distribution of wine quality?
What is the relationship between wine quality and the features in the dataset?

## Plan
- Acquire data from data.world
- Prepare data
  - Remove unnecessary features
  - Identify and replace missing values
  - Alter innapropriate data types
- Explore the data to find drivers and answer intital questions
- Apply clustering algorithms to find patterns
- Create a model to predict values
  - Use features identified in explore to build predictive models
  - Evaluate models on train and validate data
  - Select the best model based on drivers identified in exploration
  - Evaluate the best model on test data
- Conclude with recommendations and next steps

## Data Dictionary
| Feature | Definition | 
|:--------|:-----------|
| 1 - fixed acidity | a measure of low volatility acids in wine|
|2 - volatile acidity | a measure of a wines volatile or gasous acidity |
|3 - citric acid | a measure of citric acid |
|4 - residual sugar | a measure of sweetness, usually g/L |
|5 - chlorides | a measure of chlorides in wine, usually mg/L|
|6 - free sulfur dioxide | amount of sulphur dioxide ions that re not chemically bound to other chemicals, usually mg/L|
|7 - total sulfur dioxide | the total amount of sulfur dioxide (includes free sulfur dioxide), measured in mg/L |
|8 - density | the density of the wine, g/mL|
|9 - pH | the pH level |
|10 - sulphates | the amount of sulphates, mg/L |
|11 - alcohol | alcohol by volume|
|12 - quality | quality from 4-7 |
|13 - red or white | the color of the wine|


## Steps to Reproduce
1. Clone this repo
2. Download data from data.world
4. Run notebook

## Takeaways
- Classification ML algorithms are more appropriate for this dataset
- No appropriate models beat basline with current feature engineering

## Recommendations
- We do not recommend using this model in production
- We recommend spending more time on feature engineering