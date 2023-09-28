import os
import pandas as pd
import nltk
import re
from stopwords import STOP_WORDS
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk import FreqDist
from wordcloud import WordCloud
import seaborn as sns

FONT_PATH = "./font/gargi.ttf"
CSV_DIRECTORY = "./PreprocessedNepaliData/"
GRAPHICS_FOLDER = "./graphics/"
TXT_OUTPUT_DIRECTORY = "./FilteredText/"

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Gargi']

# nltk.download('stopwords')
# nltk.download('punkt')

stop_words = set(STOP_WORDS)

def is_nepali_word(word):
    return bool(re.search(r'[\u0900-\u097F]', word)) and word != "।"

def read_csv(file_path):
    data = pd.read_csv(file_path)
    return data

def save_text_to_txt(text, output_path):
    with open(output_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(text)

def extract_and_filter_text_from_csv(file_path):
    data = read_csv(file_path)
    nepali_text = ""
    if "title" in data.columns and "content" in data.columns:
        for index, row in data.iterrows():
            title = row["title"] if isinstance(row["title"], str) else ""
            content = row["content"] if isinstance(row["content"], str) else ""

            text = title + " " + content
            sentences = text.split("।")
            for sentence in sentences:
                words = nltk.word_tokenize(sentence)
                nepali_words = [word for word in words if is_nepali_word(word) and word not in stop_words]
                nepali_text += " ".join(nepali_words) + " "
    return nepali_text

def calculate_metrics_sentences(sentences):
    num_sentences = len(sentences)
    words = []

    for sentence in sentences:
        sentence_words = sentence.split()
        words.extend(sentence_words)

    avg_sentence_length = len(words) / num_sentences if num_sentences > 0 else 0
    avg_word_length = sum(len(word) for word in words) / len(words) if len(words) > 0 else 0
    vocabulary = set(words)
    vocab_size = len(vocabulary)

    return vocabulary, avg_sentence_length, avg_word_length, vocab_size

def process_filtered_csv_files_and_save_corpus(directory, output_path):
    summary = ""
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            filtered_text = extract_and_filter_text_from_csv(file_path)
            if filtered_text:
                summary += filtered_text + " । "
    save_text_to_txt(summary, output_path)

def generate_wordcloud_from_frequencies(word_freq_dist, font_path):
    wordcloud = WordCloud(width=800, height=400, background_color='white',colormap='plasma', font_path=font_path).generate_from_frequencies(word_freq_dist)
    plt.figure(figsize=(16, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    wordcloud_path = os.path.join(GRAPHICS_FOLDER, "wordcloud_nepali.png")
    plt.savefig(wordcloud_path)
    plt.show()

def plot_word_freq_dist(words):
    word_freq_dist = FreqDist(words)
    # word_freq_dist.plot(50, cumulative=False)
    print(word_freq_dist.most_common(20))
    histogram_path = os.path.join(GRAPHICS_FOLDER, "word_frequency_histogram.png")
    plt.savefig(histogram_path)
    # plt.show()
    plt.close()
    
def plot_top_n_words(words, n=20):
    word_freq_dist = FreqDist(words)
    common_words = word_freq_dist.most_common(n)
    common_words_df = pd.DataFrame(common_words, columns=['Word', 'Frequency'])
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Frequency', y='Word', data=common_words_df, palette='viridis')
    plt.title(f"Top {n} Most Frequent Words")
    plt.xlabel("Frequency")
    plt.ylabel("Word")
    plt.show()
    plt.close()

def main():
    corpus_output_path = os.path.join(TXT_OUTPUT_DIRECTORY, "corpus.txt")
    process_filtered_csv_files_and_save_corpus(CSV_DIRECTORY, corpus_output_path)
    
    with open(corpus_output_path, "r", encoding="utf-8") as corpus_file:
        summary = corpus_file.read()
        print("Corpus Length:", len(summary))
        print("Corpus:", summary[:1000])
    sentences = summary.split("।")
    
    sentence_vocabulary, avg_sentence_len, avg_word_len, sentence_vocab_size = calculate_metrics_sentences(sentences)
    print("Average Sentence Length:", avg_sentence_len)
    print("Average Word Length:", avg_word_len)
    print("Vocabulary Size (Unique Words):", sentence_vocab_size)
    print("Vocabulary:", list(sentence_vocabulary)[:10])
    
    words = nltk.word_tokenize(summary)
    
    word_freq_dist = FreqDist(words)
    
    generate_wordcloud_from_frequencies(word_freq_dist, FONT_PATH)
    plot_word_freq_dist(words)

    plot_top_n_words(words, n=20)
    
    total_sentences = len(sentences)
    print("Total Sentences:", total_sentences)
    
    print("Top Occurring Words:")
    for word, frequency in word_freq_dist.most_common(20):
        print(f"{word}: {frequency}")

if __name__ == "__main__":
    main()