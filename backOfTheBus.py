# run source venv/bin/activate before running

import pandas as pd 
import spacy as sp 


file_path = '/Users/jonnguyen/Downloads/CSVs/kaggle_bot_accounts.csv'
bot = pd.read_csv(file_path)
#print(bot.describe())
#print(bot.columns)