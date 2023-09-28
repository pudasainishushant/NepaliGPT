import os
import nltk
from nltk.corpus import stopwords
from nltk import FreqDist
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd


FONT_PATH = "/home/kushal/Videos/Corpus Preprocess/font/gargi.ttf"

# Replace with the path to the directory containing CSV files
CSV_DIRECTORY = "/home/kushal/Videos/Corpus Preprocess/PreprocessedNepaliData/"


def read_csv(file_path):
    data = pd.read_csv(file_path)
    return data

def process_csv_files(directory):
    summary = ""
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            data = read_csv(file_path)
            if "title" in data.columns and "content" in data.columns:
                data["summary"] = data["title"] + " " + data["content"]
                summary += " ".join(data["summary"].tolist()) + " "     
    return summary


def main():
    summary = process_csv_files(CSV_DIRECTORY)
    chars = sorted(list(set(summary)))
    vocab_size = len(chars)
    
    print("Unique characters found in the summary:")
    print(chars)
    print("\nVocabulary size:", vocab_size)
    
if __name__ == "__main__":
    print("Processing CSV files...")
    main()
