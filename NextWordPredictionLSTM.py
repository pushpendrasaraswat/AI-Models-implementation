import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

corpus = [

    "Deep Learning is AMAZING!!! 😍",
    "Deep-learning is Powerful...",
    "Artificial Intelligence is the Future!!",
    "Machine learning is FUN :)",
    "I LOVE Deep Learning.",
    "I love machine-learning.",
    "Python is easy!!!",
    "Python is Powerful??",
    "TensorFlow is a Deep Learning library.",
    "Keras makes Deep Learning easy.",
    "Data Science & Machine Learning!!",
    "Natural Language Processing (NLP) is Awesome.",
    "AI will change the WORLD in 2030!",
    "Deep Learning, Deep Learning, Deep Learning...",
    "Email me at abc@gmail.com",
    "Visit https://openai.com",
    "Price is ₹999 only!!!",
    "#AI #MachineLearning #DeepLearning",
    "I can't believe this works.",
    "NLP > Traditional ML ?"

]

for i, sentence in enumerate(corpus):
    print(f"Sentence {i+1}")
    print(sentence)
    print("-"*60)


lower_corpus = []

for sentence in corpus:
    sentence = sentence.lower()

    lower_corpus.append(sentence)

for sentence in lower_corpus:
    print(sentence)




import re
cleaned = []

for sentence in lower_corpus:
    sentence = re.sub(r'https?://\S+',"",sentence)
    cleaned.append(sentence)

for sentence in cleaned:
    print(sentence)




# remove email ids
temp = []
for sentence in cleaned:
    sentence = re.sub(r'\S+@\S+','',sentence)
    temp.append(sentence)

cleaned = temp

for sentence in cleaned:
    print(sentence)



# remove emojis
temp = []
for sentence in cleaned:
    sentence = re.sub(r'[^\x00-\x7F]+','',sentence)
    temp.append(sentence)

cleaned = temp

for sentence in cleaned:
    print(sentence)



# remove numbers
temp = []
for sentence in cleaned:
    sentence = re.sub(r'\d+','',sentence)
    temp.append(sentence)

cleaned = temp
for sentence in cleaned:
    print(sentence)



temp = []
for sentence in cleaned:
    sentence = sentence.replace("-"," ")
    temp.append(sentence)

cleaned = temp


# remove punctuations
import string
temp = []

for sentence in cleaned:
    sentence = sentence.translate(
        str.maketrans("","",string.punctuation)
    )
    temp.append(sentence)

cleaned = temp
for sentence in cleaned:
    print(sentence)



# remove extra spaces
final_corpus = []

for sentence in cleaned:
    sentence = re.sub(r'\s+',' ', sentence)
    sentence = sentence.strip()
    final_corpus.append(sentence)

for sentence in final_corpus:
    print(sentence)



# tokenization
tokenized_sentences = []

for sentence in final_corpus:
    words = sentence.split()
    tokenized_sentences.append(words)

print(tokenized_sentences)



# build vocabulary
vocab = set()

for sentence in tokenized_sentences:
    for word in sentence:
        vocab.add(word)

vocab = sorted(list(vocab))
print(vocab)


# word to integer mapping
word2idx = {}
idx2word = {}

for i,word in enumerate(vocab):
    word2idx[word] = i+1
    idx2word[i+1] = word

print(word2idx)


# convert sentence to numbers
numerical_sentences = []

for sentence in tokenized_sentences:
    encoded = []

    for word in sentence:
        encoded.append(word2idx[word])

    numerical_sentences.append(encoded)

print(numerical_sentences)


# creating training examples
"""
    deep learning is amazing
    deep -> learning
    deep learning -> is
    deep learning is -> amazing
"""

input_sequences = []

for sentence in numerical_sentences:
    for i in range(1,len(sentence)):

        sequence = sentence[:i+1]
        input_sequences.append(sequence)

print(input_sequences)

max_len = max(len(seq) for seq in input_sequences)
print(max_len)

# padding

padded_sequences = []

for seq in input_sequences:
    padded = [0] * (max_len - len(seq)) + seq
    padded_sequences.append(padded)

padded_sequences = np.array(padded_sequences)
print(padded_sequences)


# split x and y
X = padded_sequences[:,:-1]
y = padded_sequences[:,-1]

print(X)
print(y)


from tensorflow.keras.utils import to_categorical
# one hot encoding
vocab_size = len(vocab) + 1
y = to_categorical(y, num_classes=vocab_size)
print(y.shape)


# build LSTM
model = Sequential()

model.add(
    Embedding(
        input_dim=vocab_size,
        output_dim=32,
        input_length=max_len-1
    )
)

model.add(
    LSTM(128)
)

model.add(
    Dense(vocab_size, activation="softmax")
)

model.build(input_shape=(None,max_len-1))
model.summary()


# compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


history = model.fit(
    X,y,epochs=300,verbose=1
)


# predict next word
seed_text = "deep learning"

words = seed_text.split()

encoded = []

for word in words:
    encoded.append(word2idx[word])

encoded = [0]*(max_len-1-len(encoded)) + encoded

encoded = np.array(encoded).reshape(1,-1)
prediction = model.predict(encoded)
predicted_index = np.argmax(prediction)

print(idx2word[predicted_index])



# generate sentence
seed_text = "machine"
next_words = 4

for _ in range(next_words):
    words = seed_text.split()
    encoded = []

    for word in words:
        encoded.append(word2idx[word])

    encoded = [0]*(max_len-1-len(encoded)) + encoded

    encoded = np.array(encoded).reshape(1,-1)
    prediction = model.predict(encoded)
    predicted_index = np.argmax(prediction)

    next_word = idx2word[predicted_index]

    seed_text += " " + next_word

print(seed_text)

# predict next word
seed_text = "deep learning"

words = seed_text.split()

encoded = []

for word in words:
    encoded.append(word2idx[word])

encoded = [0]*(max_len-1-len(encoded)) + encoded

encoded = np.array(encoded).reshape(1,-1)
prediction = model.predict(encoded)
predicted_index = np.argmax(prediction)

print(idx2word[predicted_index])