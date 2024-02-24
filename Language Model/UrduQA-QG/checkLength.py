import codecs
import json

file = codecs.open("eval.json")
dataset = json.load(file)

length = 0
for data in dataset["data"]:
    for paragraph in data["paragraphs"]:
        if (len(paragraph["context"]) > length):
            length = len(paragraph["context"])
print(length)