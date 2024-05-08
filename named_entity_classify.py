import nltk
from nltk import word_tokenize, pos_tag, ne_chunk

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Define the text
text = """Apple Inc. is an American multinational technology company headquartered in Cupertino, California. It
was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976. The company designs, develops, and sells
consumer electronics, computer software, and online services. Some of its most popular products are the iPhone,
Mac computers, and Apple Watch."""

# Tokenize the text
tokens = word_tokenize(text)

# Perform part-of-speech tagging
tagged_tokens = pos_tag(tokens)

# Perform named entity recognition
named_entities = ne_chunk(tagged_tokens)

# Print named entities
for entity in named_entities:
    if hasattr(entity, 'label'):
        print(f"{entity.label():{30}} {' '.join(c[0] for c in entity)}")