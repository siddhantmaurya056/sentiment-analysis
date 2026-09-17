from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
reviews = [
    "I love this product",
    "this product is excellent",
    "very good experience",
    "amazing product",
    "i am happy with the service",
    "the service is wonderful",
    "i really enjoyed this product",
    "this is fantastic",
    "very satisfied with the purchase",
    "great product and service",

    "I hate this product",
    "this product is terrible",
    "very bad experience",
    "awful product",
    "i am disappointed with the service",
    "the service is horrible",
    "i really disliked this product",
    "this is awful",
    "very dissatisfied with the purchase",
    "poor product and service"]

# 1 = positive review, 0 = negative review
labels = [1, 1, 1, 1, 1,
          1, 1, 1, 1, 1,
          0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, ]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(reviews)

model = MultinomialNB()
# train the model with the reviews and their corresponding labels
model.fit(X, labels)