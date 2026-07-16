import pandas as pd
import matplotlib.pyplot as plt
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

from sklearn.preprocessing import LabelEncoder


"""
    Amazon recieves millions of customer complaints daily
    Instead of manually reading each complaint, we want LSTM to automatically identify the departments

    Possible departments
    1. Delivery
    2. Payments
    3. Accounts
    4. Technical support
    5. Greetings
    ...

"""

texts = [

# Delivery
"My package has not arrived",
"Delivery is delayed",
"Where is my order",
"Courier has not reached",
"My parcel is missing",
"Order status is not updated",

# Payment
"Money deducted twice",
"Payment failed",
"I need my refund",
"Refund not received",
"Double payment happened",
"Charged twice",

# Account
"Unable to login",
"Forgot my password",
"Cannot sign in",
"Password reset not working",
"My account is locked",
"Login issue",

# Technical
"Battery drains quickly",
"Phone is overheating",
"Camera quality is poor",
"Speaker not working",
"Screen has dead pixels",
"Phone hangs frequently",

# Greeting
"Hello",
"Hi",
"Good morning",
"Thank you",
"Good evening",
"How are you"

]

labels = [

"Delivery","Delivery","Delivery","Delivery","Delivery","Delivery",

"Payment","Payment","Payment","Payment","Payment","Payment",

"Account","Account","Account","Account","Account","Account",

"Technical","Technical","Technical","Technical","Technical","Technical",

"Greeting","Greeting","Greeting","Greeting","Greeting","Greeting"

]

df = pd.DataFrame(
    {
        "CustomerQuery":texts,
        "Departments":labels
    }
)

print(df)

encoder = LabelEncoder()
y=encoder.fit_transform(labels)
print(encoder.classes_)
print(y)

for i,label in enumerate(encoder.classes_):
    print(i,"->",label)


tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index
print(word_index)


sequences = tokenizer.texts_to_sequences(texts)

for i in range(5):
    print(texts[i])
    print(sequences[i])
    print()


X = pad_sequences(
    sequences,
    maxlen=6,
    padding="post"
)
print(X)

vocab_size = len(word_index) + 1
print(vocab_size)

# Build LSTM
model = Sequential()

model.add(
    Embedding(
        input_dim=vocab_size,
        output_dim=16,
        input_length=6
    )
)

model.add(
    LSTM(32)
)

model.add(
    Dense(
        len(encoder.classes_),
        activation="softmax"
    )
)


model.build(input_shape=(None,6))
model.summary()


model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
   X,
   y,
   epochs=150,
   verbose=1
)

plt.plot(history.history["accuracy"])
plt.title("Training Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()


model.save(
    "lstm-department-prediction.keras"
)

with open(
    "lstm-tokenizer.pkl",
    "wb"
) as file:

    pickle.dump(
        tokenizer,
        file
    )


print("Model and tokenizer saved successfully")