import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


MAX_LENGTH = 200


# Load already trained model
model = load_model(
    "imdb-rnn-sentiment_model.keras"
)


# Load same tokenizer used during training
with open(
    "imdb-rnn-tokenizer.pkl",
    "rb"
) as file:

    tokenizer = pickle.load(file)


review = [
    """
    This movie was absolutely fantastic.
    The acting was excellent and the story
    kept me interested until the end.
    """
]


# Text → numbers
sequence = tokenizer.texts_to_sequences(
    review
)


# Make length 200
padded_sequence = pad_sequences(
    sequence,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)


# Predict
prediction = model.predict(
    padded_sequence
)[0][0]


print("Raw prediction:", prediction)


if prediction >= 0.5:
    print("Positive sentiment")
else:
    print("Negative sentiment")