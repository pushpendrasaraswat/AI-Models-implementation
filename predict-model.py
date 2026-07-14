import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load model
model = load_model("rnn-sentiment_model.keras")


# Load tokenizer
with open("rnn-tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)


test = [
    "Movie was horrible"
]


seq = tokenizer.texts_to_sequences(test)

pad = pad_sequences(seq, maxlen=4)


prediction = model.predict(pad)


print(prediction)


if prediction[0][0] > 0.5:
    print("Positive sentiment")
else:
    print("Negative sentiment")