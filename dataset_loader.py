import nltk
from nltk.corpus import movie_reviews
import random

# Download dataset if not already present
nltk.download('movie_reviews')

def load_data():
    documents = [(list(movie_reviews.words(fileid)), category)
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]
    
    random.shuffle(documents)  # Shuffle the data for randomness
    return documents

if __name__ == "__main__":
    data = load_data()
    print(f"Loaded {len(data)} samples.")
