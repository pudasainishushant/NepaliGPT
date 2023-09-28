








import matplotlib.pyplot as plt
from wordcloud import WordCloud

font="/home/shushant/Desktop/nepaliGPT/data/font/gargi.ttf"

with open("/home/shushant/Desktop/nepaliGPT/data/processed/nepali_cc2.txt","r") as file:
    content = file.read()


# Create a WordCloud object
wordcloud = WordCloud(width=800, height=400, background_color='white',font_path=font).generate(content)



# Display the generated word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
plt.savefig("wordcloud_nepali.png")
import nltk
# nltk.download('punkt')

# Tokenize the Nepali corpus
tokens = nltk.word_tokenize(content)

# Get the total number of tokens
total_tokens = len(tokens)

print("Total number of tokens in the corpus:", total_tokens)
