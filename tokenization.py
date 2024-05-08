import nltk
from nltk.tokenize import word_tokenize

# Download the required NLTK data
nltk.download('punkt')

# Define the sample text
sample_text = "Tokenization is the process of splitting text into words or other meaningful elements."

# Tokenize the sentence
tokens = word_tokenize(sample_text)

# Print the tokens
print(tokens)