# %% [markdown]
# # Task 1: Fraud Detection

# %% [markdown]
# ## Libraries Used

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%pip install gensim
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier

# %% [markdown]
# ## Problem Description
# 
# The goal of this task is to build a model that predicts whether a given transaction is fraudulent or not using the given features. Those features include the transaction amount, the age of the account, and whether there has been a change in location for the transaction. The constraints of this problem are that the model implemented must be either a Naive Bayes or a Gaussian Discriminant Analysis model.
# 
# In developing the final model, the typical steps of the ML pipeline will be followed. This includes data preprocessing, model selection, model evaluation, and interpretation. The data preprocessing will involve the normalization of the data, the splitting of the data into training and testing sets, and the encoding of the categorical data. The model selection will involve the selection of the best model to use for the given data. The model evaluation will involve the evaluation of the model using several metrics. Finally, the interpretation will involve the interpretation of the task results and the model to analyse the most important features, potential biases, and the model's performance.

# %% [markdown]
# ## Data Collection
# 
# The data for this task is provided in the file `fraud_detection.xls`. It contains these columns:
# 
# - `Transaction_Amount`: The amount of the transaction (float).
# - `Transaction_Time`: The time of the transaction (int).
# - `Account_Age`: The age of the account (float).
# - `Previous_Location`: The last known location of the account (string).
# - `New_Location`: The location of the transaction (string).
# - `Location_Change`: Whether there has been a change in location for the transaction (binary).
# - `Is_Fraud`: Whether the transaction is fraudulent (binary).
# 
# The data will be loaded into a pandas DataFrame for further processing. Even though the file extension is `.xls`, the data is actually in the CSV format, and the `read_csv` function will be used to load the data.

# %%
primary_data = pd.read_csv('fraud_detection.xls')

# %%
primary_data.head()

# %%
primary_data.describe()

# %% [markdown]
# An initial look suggests that the `Location_Change` column is very unbalanced with the mean > standard deviation. Additionally, the lower and upper quartiles and the median are equal. The `Is_Fraud` column also appears unbalanced with all relevant quartiles and the minimum equal to each other, equal to 0, with the max being 1. The mean is also very close to 0.
# 
# This would suggest the need for outlier handling in later stages.

# %%
primary_data.info()

# %% [markdown]
# There are 10,000 rows in the dataframe, with three columns with float values, two integer columns, and 2 object columns. This is not entirelt accurate as the `Is_Fraud` column has binary values (0 and 1), but they are stored as float values (0.0 and 1.0).

# %% [markdown]
# ## Data Preprocessing

# %% [markdown]
# This section will:
# - Handle missing values in any of the columns (unlikely as the initial data exploration showed no indication of null values).
# - Handle outliers.
# - Scaling of numericaal features.
# - Encoding of categorical variables.

# %% [markdown]
# ### Imputing Missing Values

# %% [markdown]
# There is no reason to impute missing values if there are no missing values, and so the first step is checking if there are any missing values.

# %%
primary_data.isnull().sum()

# %% [markdown]
# There are no null values in the dataframe so no further operations are carried out for this section.

# %% [markdown]
# ### Handling Outliers

# %% [markdown]
# An idea that occurred to me was to handle any outlier values as they could impact the performance of the model by making it biased to certain extreme values. The approach would have been to drop rows containing outliers. However, in subsequent steps, I realised that dropping rows resulted in many of the rows that mapped to a fraudulent transaction being dropped, and so this step is skipped completely.

# %%
def handle_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    print(f'Number of outliers in {column}: {len(outliers)}')
    if len(outliers) > 0:
        print(f'Dropping outliers in {column}')
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

# %% [markdown]
# *note:* no longer relevant

# %%

# numerical_columns = ['Transaction_Amount', 'Transaction_Time', 'Account_Age']
# for column in numerical_columns:
#     primary_data = handle_outliers(primary_data, column)

# %%
primary_data.describe()

# %% [markdown]
# ### Scaling of Numerical Features

# %% [markdown]
# To determine what scaling method should be used, i.e., normalisation or standardisation, the distribution of each column must be checked, and the appropriate method used. 
# 
# To check the distribution, the skew method is being used. If the skew of a column is < 0.5, it suggests that the column is normally distributed, and so should be standardised, which is an appropriate method for normal distributions as it maintains the relationships between observations in terms of the mean and standard deviation.
# 
# If the skew of a column is greater than 0.5, it suggests that it is not normally distributed, and so normalisation is applied to the column, to ensure that the variances and patterns in the data are maintained even as the scale is reducing.
# 
# Using skewness as a measure of normality might not be pperfect but is an intuitive appraoch as an ideal normal distribution has a skewness of 0.

# %%
primary_data.dtypes

# %% [markdown]
# As already highlighted, the column `Is_Fraud` should contain integral values, not float values, and so the column is cast.

# %%
primary_data['Is_Fraud'] = primary_data['Is_Fraud'].astype(int)

# %%
primary_data.dtypes

# %%
numerical_columns = primary_data.select_dtypes(include=[float, int]).columns.drop(['Is_Fraud'])

# %%
for column in numerical_columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(primary_data[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# %%
scalars = {}

for column in numerical_columns:
    if primary_data[column].skew() < 0.5: 
        print(f'{column} is normally distributed. Standardizing.')
        primary_data[column] = (primary_data[column] - primary_data[column].mean()) / primary_data[column].std()
        scalars[column] = {'mean': primary_data[column].mean(), 'std': primary_data[column].std(), 'method': 'standardization'}
    else:
        print(f'{column} is not normally distributed. Normalizing.')
        primary_data[column] = (primary_data[column] - primary_data[column].min()) / (primary_data[column].max() - primary_data[column].min())
        scalars[column] = {'min': primary_data[column].min(), 'max': primary_data[column].max(), 'method': 'normalization'}


# %%
primary_data.describe()

# %% [markdown]
# ### Encoding Categorical Features

# %% [markdown]
# The only two categorical features are `Previous_Location` and `New_Location`. These are locations and could be as many unique values as there are rows in the dataframe, so using OneHot encoding, or even LabelEncoding would be very impractical and expensive. A better approach would be to create a word embedding for each city in the list.

# %% [markdown]
# Word vector embeddings are used to represent the `Previous_Location` and `New_Location` features in place of traditional one-hot encoding. Given that these columns could contain thousands of unique entries, one-hot encoding results in high-dimensional and sparse matrices, making it computationally expensive and potentially prone to overfitting. By using word embeddings, each unique location name is mapped to a dense, fixed-size vector that preserves semantic relationships between locations. This compact representation allows the model to capture nuanced similarities between locations, such as those that may be geographically close or frequently linked in transactions, which would be difficult to capture with one-hot encoding.
# 
# Using embeddings in this way helps reduce the complexity of the data while enhancing its expressiveness, supporting efficient training and potentially better predictive accuracy for fraud detection.

# %%
locations = set()

locations.update(primary_data['Previous_Location'].unique())
locations.update(primary_data['New_Location'].unique())

locations = list(locations)
locations

# %%
w2v_model = Word2Vec(sentences=[locations], vector_size=100, window=5, min_count=1, workers=4)
def get_embedding(location, model):
    return model.wv[location] if location in model.wv else np.zeros(model.vector_size)

# %%
primary_data['Previous_Location_Embed'] = primary_data['Previous_Location'].apply(lambda x: get_embedding(x, w2v_model))
primary_data['New_Location_Embed'] = primary_data['New_Location'].apply(lambda x: get_embedding(x, w2v_model))

# %%
primary_data

# %% [markdown]
# ## Exploratory Data Analysis

# %% [markdown]
# To provide context to some of the visualisations seen later, below are ratios of the numbers of fraudelent and non-fraudulent transactions.

# %% [markdown]
# *Fraudenlent transactions (count):*

# %%
(primary_data['Is_Fraud'] == 1).sum()

# %% [markdown]
# As a ratio of the total number of transactions:

# %%
(primary_data['Is_Fraud'] == 1).sum()/len(primary_data) * 100

# %% [markdown]
# *Non-fraudenlent transactions (count):*

# %%
(primary_data['Is_Fraud'] == 0).sum()

# %% [markdown]
# As a ratio of the total number of transactions:

# %%
(primary_data['Is_Fraud'] == 0).sum()/len(primary_data) * 100

# %% [markdown]
# The first set of relationships to be analysed is whether there appear to be relationships between each of the features, and whether a transaction was fraudulent or not.

# %% [markdown]
# ### Relationships between Features and Is_Fraud

# %% [markdown]
# #### Transaction Amount vs Is_Fraud

# %%
plt.figure(figsize=(10, 6))
sns.boxplot(x='Is_Fraud', y='Transaction_Amount', data=primary_data)
plt.title('Transaction Amount vs Is_Fraud')
plt.xlabel('Is_Fraud')
plt.ylabel('Transaction Amount')
plt.grid(True)
plt.show()

# %%
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Transaction_Amount', y='Is_Fraud', data=primary_data, hue='Is_Fraud', palette='coolwarm', alpha=0.6)
plt.title('Transaction Amount vs Is_Fraud')
plt.xlabel('Transaction Amount')
plt.ylabel('Is_Fraud')
plt.grid(True)
plt.show()

# %% [markdown]
# Almost all transactions that are fraudulent have a relatively higher transaction amount than those that are not fraudulent. This suggests that the transaction amount is a good indicator of whether a transaction is fraudulent or not.

# %% [markdown]
# #### Transaction Time vs Is_Fraud

# %%
plt.figure(figsize=(10, 6))
sns.violinplot(x='Is_Fraud', y='Transaction_Time', data=primary_data)
plt.title('Transaction Time vs Is_Fraud')
plt.xlabel('Is_Fraud')
plt.ylabel('Transaction Time')
plt.grid(True)
plt.show()

# %%
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Transaction_Time', y='Is_Fraud', data=primary_data, hue='Is_Fraud', palette='coolwarm', alpha=0.6)
plt.title('Transaction Time vs Is_Fraud')
plt.xlabel('Transaction Time')
plt.ylabel('Is_Fraud')
plt.grid(True)
plt.show()

# %% [markdown]
# For every hour, there seem to be equivalent numbers of fraudulent and non-fraudulent transactions. This suggests that the transaction time is not a good indicator of whether a transaction is fraudulent or not.

# %% [markdown]
# #### Account Age vs Is_Fraud

# %%
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Account_Age', y='Is_Fraud', data=primary_data, hue='Is_Fraud', palette='coolwarm', alpha=0.6)
plt.title('Account Age vs Is_Fraud')
plt.xlabel('Account Age')
plt.ylabel('Is_Fraud')
plt.grid(True)
plt.show()

# %% [markdown]
# All fraudulent transactions come from young accounts (the converse is not true). This is a strong indicator that account age is a highly correlated feature with fraudulent transactions.

# %% [markdown]
# #### misc

# %%
# previous_location_embeddings = np.vstack(primary_data['Previous_Location_Embed'].values)
# new_location_embeddings = np.vstack(primary_data['New_Location_Embed'].values)

# tsne = TSNE(n_components=2, random_state=42)
# previous_location_tsne = tsne.fit_transform(previous_location_embeddings)
# new_location_tsne = tsne.fit_transform(new_location_embeddings)

# previous_location_df = pd.DataFrame(previous_location_tsne, columns=['x', 'y'])
# previous_location_df['Is_Fraud'] = primary_data['Is_Fraud'].values

# new_location_df = pd.DataFrame(new_location_tsne, columns=['x', 'y'])
# new_location_df['Is_Fraud'] = primary_data['Is_Fraud'].values

# %% [markdown]
# #### Previous Location vs Is_Fraud

# %%
location_fraud_crosstab = pd.crosstab(primary_data['Previous_Location'], primary_data['Is_Fraud'])
location_fraud_crosstab.plot(kind='bar', figsize=(14, 8), colormap='coolwarm')
plt.title('Previous Location vs Is_Fraud')
plt.xlabel('Previous Location')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.legend(title='Is_Fraud')
plt.grid(True)
plt.show()

# %% [markdown]
# No region has a significantly higher number of fraudulent transactions than any of the other regions, so this is probably not a good feature for judging if a transaction is fraudulent.

# %% [markdown]
# #### New Location vs Is_Fraud

# %%
location_fraud_crosstab = pd.crosstab(primary_data['New_Location'], primary_data['Is_Fraud'])
location_fraud_crosstab.plot(kind='bar', figsize=(14, 8), colormap='coolwarm')
plt.title('New Location vs Is_Fraud')
plt.xlabel('New Location')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.legend(title='Is_Fraud')
plt.grid(True)
plt.show()

# %% [markdown]
# No region has a significantly higher number of fraudulent transactions than any of the other regions, so this is probably not a good feature for judging if a transaction is fraudulent.

# %% [markdown]
# #### Location Change vs Is_Fraud

# %%
location_fraud_crosstab = pd.crosstab(primary_data['Location_Change'], primary_data['Is_Fraud'])
location_fraud_crosstab.plot(kind='bar', figsize=(14, 8), colormap='coolwarm')
plt.title('Chnage in Location vs Is_Fraud')
plt.xlabel('Location Change')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.legend(title='Is_Fraud')
plt.grid(True)
plt.show()

# %%
((primary_data['Location_Change'] < 0) & (primary_data['Is_Fraud'] == 1)).sum()

# %%
((primary_data['Location_Change'] < 0) & (primary_data['Is_Fraud'] == 0)).sum()

# %%
((primary_data['Location_Change'] > 0) & (primary_data['Is_Fraud'] == 1)).sum()

# %%
((primary_data['Location_Change'] > 0) & (primary_data['Is_Fraud'] == 0)).sum()

# %% [markdown]
# An almost insignificant number of fraudulent transactions result from a transaction where the transaction was made from a location which was not the location where the card was issued (182/1000).
# 
# Tentatively, this feature is not a good predictor of fraudulent transactions.

# %% [markdown]
# ### Correlation Analysis

# %% [markdown]
# To support my hypotheses about which features are good/bad predictors, I will use a correlation to find the highest correlated features.

# %%
primary_data.columns.to_list()

# %%
locations = primary_data[['Previous_Location', 'New_Location']]

# %%
locations

# %%
numeric_columns = primary_data.select_dtypes(include=[float, int]).columns

corr_df = primary_data[numeric_columns].corr()

# %%
corr_df

# %%
plt.figure(figsize=(12, 8))
sns.heatmap(corr_df, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# %% [markdown]
# Transaction amount is the highest correlated feature with `Is_Fraud` with a correlation of 0.87. This is followed by `Account_Age` with a correlation of -0.13. The other features have an absolute correlation of less than 0.001.

# %% [markdown]
# ## Feature Engineering

# %% [markdown]
# Based on the relationships observed in the previous step, I will create new features that combine the most correlated features with `Is_Fraud`. I will also combine less correlated features to see if they can be used to predict `Is_Fraud`.

# %% [markdown]
# I will combine Location_Change and Transaction_Time (the 2 least correlated features) to see if together, their correlation with `Is_Fraud` is higher than their individual correlations.

# %%
primary_data['Time_And_Location'] = (primary_data['Transaction_Time'] * primary_data['Location_Change'])

# %%
numeric_columns = primary_data.select_dtypes(include=[float, int]).columns

corr_df = primary_data[numeric_columns].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_df, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# %%
corr_df

# %% [markdown]
# I will also combine Transaction_Amount and Account_Age to see if together, their correlation with `Is_Fraud` is higher than their individual correlations.

# %%
primary_data['Transaction_Amount_Acc_Age'] = primary_data['Transaction_Amount'] * -primary_data['Account_Age']

# %%
numeric_columns = primary_data.select_dtypes(include=[float, int]).columns

corr_df = primary_data[numeric_columns].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_df, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# %%
cols = primary_data[['Is_Fraud']]
primary_data.drop('Is_Fraud', axis=1, inplace=True)
primary_data = pd.concat([primary_data, cols], axis=1)
primary_data.head()

# %%
numeric_columns = primary_data.select_dtypes(include=[float, int]).columns

corr_df = primary_data[numeric_columns].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_df, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# %% [markdown]
# No feature created has a correlation with `Is_Fraud` greater than 0.87, the correlation of `Transaction_Amount` with `Is_Fraud`.

# %% [markdown]
# ## Model Selection, Development, and Evaluation

# %% [markdown]
# In accordance with the constraints of the task, the models that will be used are Naive Bayes and Gaussian Discriminant Analysis. The models will be trained on the training data and evaluated on the testing data. The model with the best performance will be selected as the final model.
# 
# Even though there are only two models to choose from, the model selection process will involve hyperparameter tuning to ensure that the best model is selected.
# 
# There are strict requirements that a distribution must be normal for the Gaussian Discriminant Analysis model to be used. Additionally, the model requires that the covariance matrix of the features be the same for all classes. This is a strong assumption and may not hold true for the data. The Naive Bayes model does not have this requirement, and so it is likely that the Naive Bayes model will be the best model to use. However, the Naive Bayes model assumes that the features are independent, which may not be true for the data especially after the feature engineering step.

# %% [markdown]
# ### Splitting the Data
# 
# I will split the data into training and testing sets using an 80/20 split. The training set will be used to train the models, and the testing set will be used to evaluate the models. The data will be split using the `train_test_split` function from the `sklearn.model_selection` module since there are no specific requirements to implement a custom split.
# 
# To ensure that the model developed is robust, I will use a KFold cross-validation with 5 folds to train the models. This will ensure that the model is trained on different subsets of the data and that the model is not overfitting the training data. This process is called cross-validation.

# %%
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score

X = primary_data.select_dtypes(include=[float, int])

X = X.drop('Is_Fraud', axis=1)
y = primary_data['Is_Fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=92)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# %% [markdown]
# ### Model Training

# %% [markdown]
# I will be using a Gaussian Discriminant Analysis model and a Naive Bayes model to train the data. The models will be trained on the training data and evaluated on the testing data. The variant of the Naive Bayes model that will be used is the Gaussian Naive Bayes model because the features are continuous.

# %% [markdown]
# #### Gaussian Naive Bayes

# %%
nb = GaussianNB()
cv_scores = cross_val_score(nb, X_train, y_train, cv=kf, scoring='accuracy')

print(f'Cross-validation scores: {cv_scores}')
print(f'Mean cross-validation score: {np.mean(cv_scores)}')

# %%
nb.fit(X_train, y_train)
test_score = nb.score(X_test, y_test)
print(f'Test score: {test_score}')

# %%
nb_test_preds = nb.predict(X_test)
nb_test_accuracy = accuracy_score(y_test, nb_test_preds)
nb_test_precision = precision_score(y_test, nb_test_preds)
nb_test_recall = recall_score(y_test, nb_test_preds)
nb_test_f1 = f1_score(y_test, nb_test_preds)
nb_test_confusion = confusion_matrix(y_test, nb_test_preds)
nb_test_classification = classification_report(y_test, nb_test_preds)

print(f'Accuracy: {nb_test_accuracy}')
print(f'Precision: {nb_test_precision}')
print(f'Recall: {nb_test_recall}')
print(f'F1 Score: {nb_test_f1}')
print(f'Confusion Matrix:\n{nb_test_confusion}')
print(f'Classification Report:\n{nb_test_classification}')

# %% [markdown]
# The model performs well on the training data with an accuracy of 0.99. The model also performs well on the testing data with an accuracy of 0.99.

# %% [markdown]
# #### Gaussian Discriminant Analysis

# %%
qda_model = QuadraticDiscriminantAnalysis()

# %%
qda_cv_scores = cross_val_score(qda_model, X_train, y_train, cv=kf, scoring='accuracy')

# %%
print(f'QDA Cross-validation scores: {qda_cv_scores}')
print(f'QDA Mean cross-validation score: {np.mean(qda_cv_scores)}')

# %%
qda_model.fit(X_train, y_train)
qda_test_pred = qda_model.predict(X_test)

# %%
qda_test_accuracy = accuracy_score(y_test, qda_test_pred)
qda_test_precision = precision_score(y_test, qda_test_pred)
qda_test_recall = recall_score(y_test, qda_test_pred)
qda_test_f1 = f1_score(y_test, qda_test_pred)

# %%
print(f'QDA Test set accuracy: {qda_test_accuracy}')
print(f'QDA Test set precision: {qda_test_precision}')
print(f'QDA Test set recall: {qda_test_recall}')
print(f'QDA Test set F1-score: {qda_test_f1}')
print(confusion_matrix(y_test, qda_test_pred))
print(classification_report(y_test, qda_test_pred))

# %% [markdown]
# ### Model Selection

# %% [markdown]
# Both models perform well on the training and testing data. The Gaussian Naive Bayes model has a slightly higher accuracy on the testing data than the Gaussian Discriminant Analysis model. The Gaussian Naive Bayes model will be selected as the final model.

# %% [markdown]
# ## Hyperparameter Tuning
# 
# Ideally, the machine learning pipeline would involve hyperparameter tuning to ensure that the best model is selected. However, the Naive Bayes and GDA models are practically perfect and so hyperparameter tuning is not necessary. Any further tuning would likely result in overfitting the model to the training data.

# %% [markdown]
# ## Feature Importance
# 
# The feature importance of the model will be analysed to determine which features are the most important in predicting whether a transaction is fraudulent or not. This will help in understanding the model and the task better. The simplest approach to this is to use a Random Forest model to determine the feature importance. This is because the Random Forest model is an ensemble model of decision trees, with each tree learning from a different subset of the data. The feature importance is calculated by averaging the importance of each feature across all the trees in the model.

# %%
rf_model = RandomForestClassifier(random_state=42)
rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=kf, scoring='accuracy')
rf_model.fit(X_train, y_train)

# %%
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
print("Feature ranking:")

for f in range(X.shape[1]):
    print(f"{f + 1}. feature {indices[f]} ({importances[indices[f]]})")

plt.figure(figsize=(10, 6))
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()

# %% [markdown]
# Testing if the Naive Bayes model performs as well or better using just the most important features.

# %%
X_train_rf = X_train[['Transaction_Amount', 'Account_Age', 'Transaction_Amount_Acc_Age']]
X_test_rf = X_test[['Transaction_Amount', 'Account_Age', 'Transaction_Amount_Acc_Age']]

model = GaussianNB()
model.fit(X_train_rf, y_train)
test_preds = model.predict(X_test_rf)
test_accuracy = accuracy_score(y_test, test_preds)
test_precision = precision_score(y_test, test_preds)
test_recall = recall_score(y_test, test_preds)
test_f1 = f1_score(y_test, test_preds)
test_confusion = confusion_matrix(y_test, test_preds)
test_classification = classification_report(y_test, test_preds)


# %%
final_scalars = {}
for column in X_train_rf.columns:
    if column in scalars:
        final_scalars[column] = scalars[column]
final_scalars['scores'] = {'accuracy': test_accuracy, 'precision': test_precision, 'recall': test_recall, 'f1': test_f1}
        

# %%
print(f'Accuracy: {test_accuracy}')
print(f'Precision: {test_precision}')
print(f'Recall: {test_recall}')
print(f'F1 Score: {test_f1}')
print(f'Confusion Matrix:\n{test_confusion}')
print(f'Classification Report:\n{test_classification}')


# %% [markdown]
# The F1-score remains the same when using just the most important features. This suggests that the model is not overfitting the data and that the model is robust.

# %% [markdown]
# ## Interpretation

# %% [markdown]
# ### Most Important Features
# 
# According to the Random Forest model, the most important feature in predicting whether a transaction is fraudulent or not is the `Transaction_Amount` feature. This is followed by the engineered feature `Transaction_Amount_Account_Age` and the `Account_Age` feature. This tallies with the correlation analysis done earlier. It futher corroborates the hypotheses derived at the exploratory data analysis stage.
# 
# Intuitively, the transaction amount is a good indicator of whether a transaction is fraudulent or not. A large amount is more likely to be fraudulent than a small amount because a bad actor would want to maximise their gains. The account age is also a good indicator of whether a transaction is fraudulent or not. A new account is more likely to be fraudulent than an old account because a bad actor would want to maximise their gains before the account is closed. The engineered feature `Transaction_Amount_Account_Age` is a good indicator of whether a transaction is fraudulent or not because it combines the two most important features.
# 
# ### Potential Biases (Overfitting)
# The model may be overfitting the training data because the model has a high accuracy on the training data and a slightly (emphasis on slightly) lower accuracy on the testing data. This suggests that the model is memorising the training data instead of generalising to new data. This could be due to the model being too complex or the model being trained on too few data points. To reduce the likelihood of overfitting, the model could be trained on more data points or the model could be made less complex.
# 
# However, this could be attributed to the imbalance in the data. The model could be biased towards the majority class, which is the non-fraudulent transactions. This could be mitigated by using a different evaluation metric, such as the F1 score, which takes into account the precision and recall of the model (both models perform well on the F1 score).
# 
# ### Improving the Model
# With the current data, the model is performing well. However, there are a few ways that the model could be improved. The first way is to collect more data. More data would help the model generalise better to new data. The second way is to engineer more features. More features would help the model learn more complex patterns in the data. The third way is to use a different model. The Naive Bayes model is performing well, but there may be other models that perform better on the data. Even a simple model like logistic regression could be used to see if it performs better than the Naive Bayes model.

# %% [markdown]
# ## Deployment

# %% [markdown]
# Intention was to deploy an application that would allow users to input the features of a transaction and get a prediction of whether the transaction is fraudulent or not. Ran out of time to do this.

# %%
import joblib
joblib.dump(model, 'model.joblib')
joblib.dump(final_scalars, 'scalars.joblib')

# %% [markdown]
# These changes were retroactively implemented in the code:
# - After scaling, the scalars (min, max, mean, std) were stored in a dictionary for use in the deployment phase.
# - The model was saved using the `joblib` library.
# - The scalars were saved using the `joblib` library.


