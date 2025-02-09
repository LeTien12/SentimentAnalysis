import json

def save_json(path_file:str , file_dataset:dict):
    with open(path_file, "w") as f:
        json.dump(file_dataset, f, indent=4)