import glob
import os
import re


directory = '/home/shushant/Desktop/nepaliGPT/data/processed'


   
txt_files = glob.glob(directory+"/"+ "/*.txt")


print(txt_files)

for text_file in txt_files:
    with open(text_file,"r") as file:
        text = file.read()
    with open("nepali_text_corpus.txt","a") as f:
        f.write(text)