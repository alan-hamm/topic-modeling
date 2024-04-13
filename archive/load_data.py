from pathlib import Path import json

data_path = Path("\\cdc.gov\CSP_Private\M728\pqn7\prodigy_poc")
for file_path in data_path.iterdir():  # iterate over directory
    lines = Path(file_path).open("r", encoding="utf8")  # open file
    for line in lines:
        task = {"text": line}  # create one task for each line of text
        print(json.dumps(task))  # dump and print the JSON