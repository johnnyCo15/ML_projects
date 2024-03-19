import pandas as pd 

# save file path
melbourne_file_path = '/Users/jonnguyen/Downloads/GITHUB/ML_projects/melb_data.csv'

# read/ store the data 
melbourne_data = pd.read_csv(melbourne_file_path)

# print summary of data
print(melbourne_data.describe())

