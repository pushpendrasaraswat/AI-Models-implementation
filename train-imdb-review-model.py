import os
import pickle
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping

# =========================================================
# 1. CONFIGURATION
# =========================================================

DATASET_PATH = "aclImdb"

VOCAB_SIZE = 10000
MAX_LENGTH = 200
EMBEDDING_DIM = 64
RNN_UNITS = 32
EPOCHS = 5


# =========================================================
# 2. FUNCTION TO READ REVIEWS
# =========================================================

def load_reviews(folder_path, label):
    reviews = []
    labels = []

    for filename in os.listdir(folder_path):

        if filename.endswith(".txt"):

            file_path = os.path.join(folder_path, filename)

            with open(file_path, "r", encoding="utf-8") as file:
                review = file.read()

                reviews.append(review)
                labels.append(label)

    return reviews, labels

# =========================================================
# 3. LOAD TRAINING DATA
# =========================================================

train_positive_reviews, train_positive_labels = load_reviews(
    os.path.join(DATASET_PATH, "train", "pos"),
    1
)

train_negative_reviews, train_negative_labels = load_reviews(
    os.path.join(DATASET_PATH, "train", "neg"),
    0
)

train_sentences = (
    train_positive_reviews
    + train_negative_reviews
)
train_labels = (
    train_positive_labels
    + train_negative_labels
)


# =========================================================
# 4. LOAD TEST DATA
# =========================================================

test_positive_reviews, test_positive_labels = load_reviews(
    os.path.join(DATASET_PATH, "test", "pos"),
    1
)

test_negative_reviews, test_negative_labels = load_reviews(
    os.path.join(DATASET_PATH, "test", "neg"),
    0
)

test_sentences = (
    test_positive_reviews
    + test_negative_reviews
)

test_labels = (
    test_positive_labels
    + test_negative_labels
)


print("Training reviews:", len(train_sentences))
print("Testing reviews:", len(test_sentences))


# =========================================================
# 5. CREATE TOKENIZER
# =========================================================

tokenizer = Tokenizer(
    num_words=VOCAB_SIZE,
    oov_token="<OOV>" # if word is not in vocabulary out of vocabulary
)


# IMPORTANT:
# Learn vocabulary ONLY from training data.
tokenizer.fit_on_texts(train_sentences)


# =========================================================
# 6. CONVERT TRAINING TEXT TO NUMBERS
# =========================================================

train_sequences = tokenizer.texts_to_sequences(
    train_sentences
)

X_train = pad_sequences(
    train_sequences,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

# =========================================================
# 7. CONVERT TEST TEXT TO NUMBERS
# =========================================================

test_sequences = tokenizer.texts_to_sequences(
    test_sentences
)

X_test = pad_sequences(
    test_sequences,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)


Y_train = np.array(train_labels)
Y_test = np.array(test_labels)

# =========================================================
# SHUFFLE TRAINING DATA
# =========================================================

# Create indexes: [0, 1, 2, 3, ...]
indices = np.arange(len(X_train))

# Randomly shuffle those indexes
np.random.shuffle(indices)

# Apply the same shuffled order to X and Y
X_train = X_train[indices]
Y_train = Y_train[indices]

print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)

print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)


# =========================================================
# 8. CREATE RNN MODEL
# =========================================================

model = Sequential()

model.add(
    Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBEDDING_DIM,
        mask_zero=True
    )
)

model.add(
    SimpleRNN(RNN_UNITS)
)

model.add(
    Dense(
        1,
        activation="sigmoid"
    )
)

model.build(input_shape=(None,MAX_LENGTH))
# =========================================================
# 9. COMPILE MODEL
# =========================================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)
# =========================================================
# 10. TRAIN MODEL
# =========================================================

model.fit(
    X_train,
    Y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping]
)
# =========================================================
# 11. TEST ON UNSEEN DATA
# =========================================================

test_loss, test_accuracy = model.evaluate(
    X_test,
    Y_test
)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# =========================================================
# 12. SAVE MODEL
# =========================================================

model.save(
    "imdb-rnn-sentiment_model.keras"
)



# =========================================================
# 13. SAVE TOKENIZER
# =========================================================

with open(
    "imdb-rnn-tokenizer.pkl",
    "wb"
) as file:

    pickle.dump(
        tokenizer,
        file
    )


print("Model and tokenizer saved successfully")

