import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
""" 
# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet') """

# Define the text
text = """Apple Inc. is an American multinational technology company headquartered in Cupertino, California. It
was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976. The company designs, develops, and sells
consumer electronics, computer software, and online services. Some of its most popular products are the iPhone,
Mac computers, and Apple Watch."""

# Tokenize the text
tokens = word_tokenize(text)

# Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

# Perform stemming
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

# Perform lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

# Print the results
print("Original Tokens:", tokens)
print("\nFiltered Tokens (no stopwords):", filtered_tokens)
print("\nStemmed Tokens:", stemmed_tokens)
print("\nLemmatized Tokens:", lemmatized_tokens)