import os
import sys
import requests

headers = {"Authorization": f"Bearer {os.environ['HUGGING_FACE_TOKEN']}"}
API_URL = f"https://datasets-server.huggingface.co/size?dataset={sys.argv[1]}"

print(API_URL)


def query():
    response = requests.get(API_URL, headers=headers)
    return response.json()


data = query()
print(data["size"]["dataset"]["num_rows"])
