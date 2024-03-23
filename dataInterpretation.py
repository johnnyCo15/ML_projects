import pandas as pd 

# run source venv/bin/activate before running

# save file path
melbourne_file_path = '/Users/jonnguyen/Downloads/GITHUB/ML_projects/melb_data.csv'

# read/ store the data 
melbourne_data = pd.read_csv(melbourne_file_path)

# print summary of data
print(melbourne_data.describe(), file = open("file.txt", "w"))

# finding avg year built (rounded to nearest integer)
avg_yr = melbourne_data.YearBuilt.mean()
avg_yr = avg_yr.round()
print("avg lot sz: ", avg_yr)

# finding avg building area 
avg_lot_sz = melbourne_data.BuildingArea.mean()
avg_lot_sz = avg_lot_sz.round()
print("avg building area: ", avg_lot_sz)

# finding newest home built 
newest_house = 2024 - melbourne_data.YearBuilt.max()
print("newest house [yrs]: ", newest_house)

# # print the columns 
# print(melbourne_data.columns)

# dropping values
# dropna drops missing values (think of na as "not available")
melbourne_data = melbourne_data.dropna(axis = 0)

# prediction target
y = melbourne_data.Price

# choosing features 
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]

print(X.describe())
print(X.head())

# building the model 
from sklearn.tree import DecisionTreeRegressor

# define model 
# specifying number to ensure same result for each run
melbourne_model = DecisionTreeRegressor(random_state = 1)

# fit model 
melbourne_model.fit(X, y)

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))
print("---------------------------------")
print(melbourne_model.predict(X))

# model validation 
from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(X)
print(mean_absolute_error(y, predicted_home_prices))


from sklearn.model_selection import train_test_split

# split data into training and validation data 
# split based on rand num generator
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# define the model 
melbourne_model = DecisionTreeRegressor()

# fit model 
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data 
val_predictions = melbourne_model.predict(val_X)
print("value prediction: ", val_predictions)

print("-----------------------------------------")
val_mae = mean_absolute_error(val_y, val_predictions)
print(val_mae)



