import json

def read_json(filename):
    with open(filename, "r") as file:
        data = json.loads(file.read())
    return data

def save_json(filename, str_json):
    with open(filename, 'w') as file:
        file.write(str_json)