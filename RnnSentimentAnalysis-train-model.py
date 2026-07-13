import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense,SimpleRNN
import pickle

#At a high level, program does this:
# Text Sentences
#        |
#        ↓
# Tokenization (words → numbers)
#        |
#        ↓
# Padding (make all sentences same length)
#        |
#        ↓
# Embedding (numbers → meaningful vectors)
#        |
#        ↓
# SimpleRNN (learn sequence patterns)
#        |
#        ↓
# Dense + Sigmoid (positive/negative prediction)
#        |
#        ↓
# Training
#        |
#        ↓
# Save model
# these are the sentence on which are model is going to train
sentences = [
    "food was amazing",
    "movie was fantastic",
    "service was excellent",
    "food was horrible",
    "movie was boring",
    "film was terribly boaring"
]
print(sentences)
# these are the actual output for each sentence , 1 - positive , 0 - negative
labels=[1,1,1,0,0,0]
print(labels)

#Tokenizer is a TensorFlow utility that converts words into numbers.
tokenizer = Tokenizer()
#This scans all sentences and creates a vocabulary.
tokenizer.fit_on_texts(sentences)
# This is the vocabulary
# was        -> 1
# food       -> 2
# movie      -> 3
# amazing    -> 4
# fantastic  -> 5
# service    -> 6
# excellent  -> 7
# horrible   -> 8
# boring     -> 9
# film       -> 10
# terribly   -> 11
# boaring    -> 12

word_index = tokenizer.word_index
# every word is matched to one index
print(word_index)
# {'was': 1, 'food': 2, 'movie': 3, 'amazing': 4, 'fantastic': 5, 'service': 6, 'excellent': 7, 'horrible': 8, 'boring': 9}

sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)
# [[2, 1, 4], [3, 1, 5], [6, 1, 7], [2, 1, 8], [3, 1, 9], [10, 1, 11, 12]]

# one vector is of 4 length and other are 3 , to make them same size we are doing padding , so all are of same size
# to add padding at the end we use pad_sequences(sequences, maxlen=4, padding='post')
#"Make every sentence length 4"
X= pad_sequences(sequences, maxlen=4)
print(X)

Y=np.array(labels)
print(Y)


model = Sequential()
#Sequential means:
#Layers will execute one after another.
# Input
#  |
# Embedding
#  |
# SimpleRNN
#  |
# Dense
#  |
# Output


model.add(
    Embedding(
        input_dim=len(word_index) + 1,#index start with 0 so +1
        output_dim=8
    )
)
#This is the most important NLP concept. Currently:

# food = 2
# movie = 3
# amazing = 4
#
# Numbers have no meaning.
# Example: food = 2 ,  horrible = 8
#The model does not know:
#food ≠ horrible
#Embedding converts numbers into vectors.
#Example: Before: food = 2
#After embedding:
# food =
# [
#     0.21,
#     0.55,
#     -0.12,
#     0.76,
#     ...
# ]
# 8 numbers because: output_dim=8 So every word becomes an 8-dimensional vector.

model.add(
    SimpleRNN(16)
)
#RNN remembers previous words.It maintains memory:Current word + Previous words, Create 16 memory units
# 16 numbers representing the sentence meaning

model.add(
    Dense(1,activation='sigmoid') # sigmoid for 0 and 1 , positive and negative sentiment
)
#This is the final classifier. It gives one output:
#0.0 -------- 1.0
#Example: 0.91 means: 91% positive
#0.12 means: 12% positive or Negative
model.build(input_shape=(None,4))
# build model
model.summary()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X,
    Y,
    epochs=75
)

# Save trained model
model.save("sentiment_model.keras")

# Stores:
# # Neural network architecture
# # Learned weights
# Configuration
#
# Next time: No training required.
with open("tokenizer.pkl", "wb") as file:
    pickle.dump(tokenizer, file)


print("Model training completed and saved")