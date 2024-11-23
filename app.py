import pickle
from flask import Flask, render_template, request

# Load the saved model and vectorizer
model = pickle.load(open('sentiment_analysis_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    tweet_text = request.form['tweet']

    # Transform the tweet using the vectorizer (ensure it was fitted before)
    tweet_features = vectorizer.transform([tweet_text])

    # Predict sentiment using the loaded model
    prediction = model.predict(tweet_features)

    # Return the sentiment result
    if prediction == 1:
        prediction_text = 'Positive Tweet'
    else:
        prediction_text = 'Negative Tweet'

    return render_template('index.html', prediction_text=prediction_text)


if __name__ == "__main__":
    app.run(debug=True)

