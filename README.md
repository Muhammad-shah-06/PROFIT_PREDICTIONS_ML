# PROFIT_PREDICTIONS_ML

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor,plot_tree
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score,KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer,mean_squared_error,r2_score,mean_absolute_error
import joblib
#load dataset

df=pd.read_csv("SampleSuperstore.csv")
print(df.head())
print(df.shape)
#df.set_option('display.max_columns',None)
print(df.info())
"""
print(df['Ship Mode'].value_counts())
print(df['Segment'].value_counts())
print(df['City'].value_counts())
print(df['State'].value_counts())
print(df['Region'].value_counts())
print(df['Category'].value_counts())
print(df['Sub-Category'].value_counts())
print(df['Sales'].value_counts())
print(df['Quantity'].value_counts())
print(df['Discount'].value_counts())

print(df['Ship Mode'].unique())
print(df['Segment'].unique())
print(df['Country'].unique())
#print(df['City'].unique())
#print(df['State'].unique())
print(df['Region'].unique())
print(df['Category'].unique())
"""
#check null values

df.info()
print(df.isnull().sum())

# check and remove duplicated values

df.duplicated().sum()
df=df.drop_duplicates(keep='first')
df.duplicated().sum()

#Remove independent column

print(df.drop(['Postal Code'],axis=1,inplace=True))
print(df.info())
print(df.shape)




#Group by Region
print(df.groupby('Region').count())

          #Most orders are from West region
         
# Plot the ship mode count
ship_mode_count = df['Ship Mode'].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=ship_mode_count.index, y=ship_mode_count.values)
#sns.barplot(x=ship_mode_count.index, y=df['Profit'].values)
plt.title("Ship Mode Count")
plt.xlabel("Ship Mode")
plt.ylabel("Count")
plt.show()

# Plot the segment count
segment_count = df['Segment'].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=segment_count.index, y=segment_count.values)
plt.title("Segment Count")
plt.xlabel("Segment")
plt.ylabel("Count")
plt.show()

# Plot the category count
category_count = df['Category'].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=category_count.index, y=category_count.values)
plt.title("Category Count")
plt.xlabel("Category")
plt.ylabel("Count")
plt.show()

# Plot the profit by state
state_profit = df.groupby('State')['Profit'].sum()
plt.figure(figsize=(12, 6))
sns.barplot(x=state_profit.index, y=state_profit.values)
plt.title("Profit by State")
plt.xlabel("State")
plt.ylabel("Profit")
plt.xticks(rotation=90)
plt.show()

#print(df.groupby('State')['Profit'].sum())

# Plot the discount by quantity
#quantity_discount = df.groupby('Quantity')['Discount'].mean()
"""plt.figure(figsize=(10, 6))
sns.barplot(x=df['Quantity'], y=df['Discount'])
plt.title("Discount by Quantity")
plt.xlabel("Quantity")
plt.ylabel("Discount")
plt.show()
"""
#plt.figure(figsize=(10, 6))
sns.relplot(x='Quantity', y='Discount', data=df, kind='line')
plt.title("Discount by Quantity")
plt.xlabel("Quantity")
plt.ylabel("Discount")
plt.show()

# Plot the categories with the highest quantity bought in California and New York, color coded by sub-categories
category_subcategory_quantity = df[df['State'].isin(['California', 'New York'])].groupby(['Category', 'Sub-Category'])['Quantity'].sum()
plt.figure(figsize=(10, 6))
sns.barplot(x=category_subcategory_quantity.index.get_level_values(0), y=category_subcategory_quantity.values, hue=category_subcategory_quantity.index.get_level_values(1))
plt.title("Categories with the Highest Quantity Bought in California and New York")
plt.xlabel("Category")
plt.ylabel("Quantity")
plt.xticks(rotation=90)
plt.legend(title="Sub-Category")
#plt.legend(loc='center left')
plt.show()

from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
df[['Ship Mode','Segment','Country','City','State','Region','Category','Sub-Category']]=df[['Ship Mode','Segment','Country','City','State','Region','Category','Sub-Category']].apply(label_encode.fit_transform)



#drop country
print(df['Country'].unique())
print(df.drop(['Country'],axis=1,inplace=True))
print(df.info())
"""
print("sales:",df['Sales'].max())
print("discount:",df['Discount'].min())
print("sub-c:",df['Sub-Category'].max())
print("qun:",df['Quantity'].max())
print("city:",df['City'].max())
print("category:",df['Category'].max())
print("region:",df['Region'].max())
print("state:",df['State'].max())
print("ship_mode:",df['Ship Mode'].max())
print("segment:",df['Segment'].max())
"""
#heatmap
plt.figure(figsize=(10, 10))
correlation=df.corr(numeric_only=True)
sns.heatmap(correlation, annot=True, cmap='Purples')
plt.show()

#label Encoding
onehot_encode=OneHotEncoder()
o_oh=onehot_encode.fit(df)
tr=o_oh.transform(df)
print(tr)

#box plot 
df.plot(kind='box')
plt.xticks(rotation=90)
plt.show()

#outlier removing
outliers_columns=["Sales","Category","Ship Mode","Quantity","Discount"]
for column in outliers_columns:
 if df[column].dtype in ["int64","float64"]: 
  Q1=df[column].quantile(0.25)
  Q3=df[column].quantile(0.75)
  iqr=Q3-Q1
  lower_bound=Q1-1.5*iqr
  upper_bound=Q3+1.5+iqr
  df=df[(df[column]>=lower_bound) & (df[column]<=upper_bound)]
  print(df)

df.plot(kind='box')
plt.xticks(rotation=90)
plt.show()

#model comparison
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
print(X)
print(y)
models=[]
models.append(('KNR',KNeighborsRegressor()))
models.append(('LNG',LinearRegression()))
models.append(('R',Ridge()))
models.append(('L',Lasso()))
models.append(('DTR',DecisionTreeRegressor()))
models.append(('RFR',RandomForestRegressor()))
models.append(('SVR',SVR()))
results=[]
names=[]
scoring='neg_mean_squared_error'
kfold=KFold(n_splits=10,shuffle=True,random_state=42)
#cv_results=cross_val_score(models,X,y,cv=kfold,scoring=scoring)
for name,model in models:
	cv_results=cross_val_score(model,X,y,cv=kfold,scoring=scoring)
	results.append(cv_results)
	names.append(name)
	print('cv_results:',cv_results)
	print(f"MSE of {name} : {cv_results.mean()}")
	
fig=plt.figure()
fig.suptitle("Algorithm Comparison")
ax=fig.add_subplot(111)
ax.set_xticklabels(names)
#ax.set_ylim(0,1)

plt.boxplot(results)
plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

predictions = rf_regressor.predict(X_test)
print(predictions)

mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')
r2 = r2_score(y_test, predictions)
print(r2)



#scatter plot of true value vs predicted value
plt.scatter(y_test, predictions, alpha=0.5,label='Perfect Prediction')
#plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), '--k', label='Perfect Prediction') 
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True vs Predicted Values')
plt.legend()
plt.show()

# Get feature importances
feature_importances = rf_regressor.feature_importances_
#feature_names = [f"Feature {i}" for i in range(len(feature_importances))]
feature_names = X.columns
print(feature_names)
# Sort features based on importance
sorted_indices = np.argsort(feature_importances)[::-1]
print("sorted columns:",sorted_indices)

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importances)), feature_importances[sorted_indices], align="center")
plt.xticks(range(len(feature_importances)), np.array(feature_names)[sorted_indices], rotation=45)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importance in Random Forest Regressor")
plt.show()


# Define the hyperparameter grid to search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
# Fit the model to the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print(f'Best Hyperparameters: {best_params}')

# Get the best model
best_model = grid_search.best_estimator_
# Evaluate the model using mean squared error
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error on Test Set: {mse}')

# Define a scoring metric (e.g., mean squared error)
scoring_metric = make_scorer(mean_squared_error, greater_is_better=False)
# Perform cross-validation
cv_scores = cross_val_score(rf_regressor, X, y, cv=5, scoring=scoring_metric)
# Display the cross-validation scores
print(f'Cross-Validation Scores: {cv_scores}')

# Load the trained model from the file
#loaded_model = joblib.load('trained_model.joblib')
new_data =[[23000,0,17,18,450,0,1,40,0,0]]
new_predictions = rf_regressor.predict(new_data)
print(new_predictions)









