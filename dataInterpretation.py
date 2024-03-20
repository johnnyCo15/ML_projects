import pandas as pd 

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