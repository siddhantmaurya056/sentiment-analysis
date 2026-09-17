from flask import Flask, render_template, request
from model import model, vectorizer
flask_app = Flask(__name__)
@flask_app.route('/')
def home():
    return render_template('index.html')
@flask_app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review'] 
    review_vector = vectorizer.transform([review])
    prediction = model.predict(review_vector)
    if prediction[0] == 1:
        sentiment = "Positive Review"
    else:
        sentiment = "Negative Review"
    probability = model.predict_proba(review_vector)[0][prediction[0]]
    return render_template('index.html', review=review, sentiment=sentiment, probability=round(probability,2))
if __name__ == '__main__':
    flask_app.run(debug=True)