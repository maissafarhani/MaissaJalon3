# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import streamlit as st
import nltk
import pickle
from textblob import TextBlob
from PIL import Image
nltk.download('brown')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


with open('./vectorizer_Model.pkl', 'rb') as vectorizer_Model:
    vec = pickle.load(vectorizer_Model)
print(vec)
with open('./NMF_Model.pkl', 'rb') as NMF_Model:
    nmf = pickle.load(NMF_Model)
print(nmf)


dict = {
    0 : "bad phone call to customer",
    1 : "bad taste",
    2 : "bad pizza",
    3 : "wrong order, bad dilivery",
    4 : "bad or slow service food",
    5 : "bad service",
    6 : "bad burgers",
    7 : "long wait",
    8 : "bad food",
    9 : "bad service in bar",
    10 : "problem with the delivery ",
    11 : "bad experience many times",
    12 : "food is not served with orgonazation",
    13 : "bad sushis",
    14 : "bad place"
}

def prediction(input : str,nb_features:int,dict=dict, blob=True) :
    liste = []
    if blob == True:
        blob = TextBlob(input)
        polarity = blob.sentiment.polarity
        print(polarity)
        if polarity > 0:
            print("POLARITE:", polarity, " ( C'EST UN COMMENTAIRE POSITIF)")

        if polarity < 0:
            print("POLARITE:", polarity, " (COMMENTAIRE NEGATIF)")
            input = [input]
            X = vec.transform(input)
            nmf_features = list(list(nmf.transform(X))[0])

            a = nmf_features
            b = []
            b.append(nmf_features.index(max(a)))
            a.pop(b[0])

            for i in range(nb_features - 1):
                index = nmf_features.index(max(a))
                b.append(index)
                a.pop(index)

            for i in b:
                liste.append(dict[i])

    return polarity, liste


#########################################################################################"




#########################################################################################
st.title("welcome to my app")
image = Image.open('maissapic.png')

st.image(image, caption='topic modeling')
text=st.text_input("enter your comment")
number = st.number_input('Insert a number of topics',min_value=1,max_value=15,step=1)

p,b=prediction(text, number, dict)
if st.button('Predict'):
    st.write(p)
    st.write(b)

