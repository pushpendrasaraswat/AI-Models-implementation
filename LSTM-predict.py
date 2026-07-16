import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import LabelEncoder

MAX_LENGTH = 200


# Load already trained model
model = load_model(
    "lstm-department-prediction.keras"
)



# Load same tokenizer used during training
with open(
    "lstm-tokenizer.pkl",
    "rb"
) as file:

    tokenizer = pickle.load(file)
labels = [

"Delivery","Delivery","Delivery","Delivery","Delivery","Delivery",

"Payment","Payment","Payment","Payment","Payment","Payment",

"Account","Account","Account","Account","Account","Account",

"Technical","Technical","Technical","Technical","Technical","Technical",

"Greeting","Greeting","Greeting","Greeting","Greeting","Greeting"

]

encoder = LabelEncoder()
y=encoder.fit_transform(labels)
print(encoder.classes_)
print(y)

def predict_department(sentence):
    seq = tokenizer.texts_to_sequences([sentence])
    pad = pad_sequences(seq, maxlen=6,padding="post")
    prediction = model.predict(pad,verbose=1)[0]
    predicted_class = np.argmax(prediction)
    department = encoder.inverse_transform([predicted_class])[0]
    confidence = prediction[predicted_class] * 100

    print('-'*50)
    print("Customer Query")
    print(sentence)
    print()
    print("Predicted Department")
    print(department)
    print()
    print(f"Confidence: {confidence:.2f}%")
    print('-'*50)


predict_department("Money deducted twice")

predict_department(
    "My package delivery was ontime but payment failed"
)


def explain_prediction(sentence):
    seq = tokenizer.texts_to_sequences([sentence])
    pad = pad_sequences(seq,maxlen=6,padding="post")
    prediction = model.predict(pad,verbose=1)[0]

    print(sentence)
    print()

    for label,prob in zip(encoder.classes_, prediction):
        print(f"{label:12s} : {prob*100:.2f}%")



explain_prediction(
    "refund not recieved"
)