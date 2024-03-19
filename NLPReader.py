import spacy

# loads spacy
nlp = spacy.load('en_core_web_trf')
locations = []

# asks user for file path
def file_finder():
    file_path = input("Enter the path of txt file: ")
    return file_path

# read lines from txt file + create dict using line num as keys
def file_opener(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    text_dict = {i + 1: line.strip() for i, line in enumerate(lines)}
    return text_dict

# 

def main():
    return 

if __name__ == "__main__":
    main()