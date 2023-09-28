import requests
from time import perf_counter
start = perf_counter()
def translate(src, trg, word):
    url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={src}&tl={trg}&dt=t&dt=bd&dj=1&q={word}"
    response = requests.get(url)
    if response.ok:
        result = response.json()
        return result["sentences"][0]["trans"]
    else:
        print("Translation request failed with status code:", response.status_code)
        return None

with open("qa.txt","r") as file:
    data = file.readlines()
translated_data = []
for i,qa in enumerate(data):
    qa_data_dict = dict()
    try:
        question,answer = qa.split("~")
        translated_question= translate(src="en", trg="ne", word=question)
        translated_answer = translate(src="en",trg="ne",word = answer)
        qa_data_dict["input"] = translated_question
        qa_data_dict["output"] = translated_answer
        print("Done for {} data",i)
    except:
        print(" ------------------ Error for {} data,i")

    translated_data.append(qa_data_dict)
import json
with open('dataset/translated_qa_final.json', 'w') as fp:
    json.dump(translated_data, fp, indent=4, ensure_ascii=False)