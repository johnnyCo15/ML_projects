# run source venv/bin/activate before running

import pandas as pd 
import spacy as sp 

file_path = '/Users/jonnguyen/Downloads/kaggle_bot_accounts.csv'
bot = pd.read_csv(file_path)
print(bot.describe())