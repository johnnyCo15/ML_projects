import pandas as pd 

def file_path_getter():
    file_path = input("Enter the path of txt file: ")
    return file_path

def file_reader(file_path):
    try: 
        with open (file_path, 'r') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None
    except Exception as e: 
        print(f"Error has occurred while reading file: (e)")
        return None

def main():
    file_path = file_path_getter()
    file_content = file_reader(file_path)
    print(file_content)

if __name__ == "__main__":
    main()
