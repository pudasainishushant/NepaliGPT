import re
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already downloaded
# nltk.download('stopwords')
# nltk.download('punkt')

# Get the Nepali stopwords from NLTK
stop_words = set(stopwords.words('nepali'))

def is_nepali_word(word):
    return bool(re.search(r'[\u0900-\u097F]', word))

def remove_english_words(text):
    english_word_pattern = r'\b[a-zA-Z]+\b'
    cleaned_text = re.sub(english_word_pattern, '', text)
    return cleaned_text

def tokenize(text):
    text_without_english = remove_english_words(text)
    words = nltk.word_tokenize(text_without_english)

    # Filter out punctuation tokens and select only tokens with Nepali characters
    nepali_words = [
        word for word in words
        if (word not in stop_words) and is_nepali_word(word) and any(c.isalpha() for c in word)
    ]

    return nepali_words

def calculate_metrics(text):
    sentences = text.split("।")
    words = text.split()

    # Remove empty strings from the words list
    words = [word for word in words if word.strip() != ""]

    # Calculate the number of sentences
    num_sentences = len(sentences) - 1

    avg_sentence_length = len(words) / num_sentences

    avg_word_length = sum(len(word) for word in words) / len(words)

    # Get unique words and their count
    vocabulary = set(words)
    vocab_size = len(vocabulary)

    return vocabulary, avg_sentence_length, avg_word_length, vocab_size
