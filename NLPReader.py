import spacy
import pandas as pd

# run source venv/bin/activate before running

# asks user for file path
def file_finder():
    file_path = input("Enter the path of txt file: ")
    return file_path

# read lines from txt file + create dict using line num as keys
def file_opener(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return {i + 1: line.strip() for i, line in enumerate(lines)}
  
# extract locations from text
def extract_location(text_dict):
    nlp = spacy.load('en_core_web_trf')
    locations = []
    for _, text in text_dict.items():
        doc = nlp(text)
        location_entities = [ent.text for ent in doc.ents if ent.label_ == "GPE" or ent.label_ == "LOC"]
        location_str = ', '.join(location_entities)
        locations.append(location_str)
    return locations

# create a dataframe from dict with key and value cols 
def create_df(text_dict, locations):
    df = pd.DataFrame(list(text_dict.items()), columns = ['key', 'value'])
    df['locations'] = locations 
    return df

# main
def main():

    file_path = file_finder()
    text_dict = file_opener(file_path)
    locations = extract_location(text_dict)
    df = create_df(text_dict, locations)

    try:
        file = open("nlp_output.txt", "x")
    except:
        file = open("nlp_output.txt", "w")
    
    file.write(str(df))
    file.close


if __name__ == "__main__":
    main()