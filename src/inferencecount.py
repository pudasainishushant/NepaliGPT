# Function to calculate the average sentence length, average word length, and vocabulary (unique words).
def calculate_metrics(text):
    sentences = text.split("।")  # Split the text into a list of sentences.
    words = text.split()  # Split the input text into a list of separate words

    # Remove empty strings from the words list
    words = [word for word in words if word.strip() != ""]

    if sentences[len(sentences) - 1] == "":  # If the last value in sentences is an empty string
        avg_sentence_length = len(words) / (len(sentences) - 1)
    else:
        avg_sentence_length = len(words) / len(sentences)

    avg_word_length = sum(len(word) for word in words) / len(words)

    # Get unique words and their count
    vocabulary = set(words)
    vocab_size = len(vocabulary)

    return avg_sentence_length, avg_word_length, vocab_size  # Return calculated metrics

text = """
कोशी प्रदेश प्रहरी कार्यालयले आर्थिक वर्ष २०७९/८० मा प्रदेश भरिमा विभिन्न घटनाका आठ हजार सात सय १९ मुद्दा दर्ता भई दुई  हजार सात सय ७३ फर्छ्यौट गरेको छ ।
सो अवधिमा रु ५२ करोड १७ लाख ७८ हजार राजस्व सङ्कलन भएको प्रदेश प्रहरी प्रमुख प्रहरी नायब महानिरीक्षक राजेशनाथ बाँस्तोलाले वार्षिक कार्यसम्पादन समीक्षा प्रगति विवरण प्रस्तुत गर्दै जानकारी दिएका हुन ।
गत आवमा आठ हजार सात सय १९ विभिन्न अपराधसम्बन्धी मुद्दा दर्ता भई कानूनी कारबाही भइरहेको तथा फैसला कार्यान्वयन तर्फ विभिन्न अपराधमा सजाय एवं जरिवाना फैसला भई फरार रहँदै आएका एक हजार चार सय ६६ जना फरार प्रतिवादी पक्राउ गरिएको र रु चार करोड २४ लाख १२ हजार राजस्व सङ्कलन भएको प्रहरी प्रमुख बाँस्तोलाले जानकारी दिए ।
"""

avg_sentence_len, avg_word_len, vocab_size = calculate_metrics(text)  # Function call
print("Average Sentence Length:", avg_sentence_len)
print("Average Word Length:", avg_word_len)
print("Vocabulary Size (Unique Words):", vocab_size)
