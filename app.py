import streamlit as st
import pickle

from keras.saving.save import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

trunc_type = 'post'
padding_type = 'post'
threshold = 0.5
size_vocabulary = 1000
oov_token = "<OOV>"
max_len = 189

model = load_model('My_model.h5')
tokenizer = pickle.load(open('My_model.pkl','rb'))

st.title("SMS Spam Classifier")
#
input_sms = st.text_input("Enter the message")

if st.button('Predict'):

    message_example_tp = pad_sequences(tokenizer.texts_to_sequences([input_sms]),
                                       maxlen=max_len,
                                       padding=padding_type,
                                       truncating=trunc_type)

    pred = float(model.predict(message_example_tp))
    if (pred > threshold):
        st.header("This message is a real text")
    else:
        st.header("This message is a spam message")


