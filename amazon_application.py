from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import string
import pandas as pd
import re
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import spacy
nlp=spacy.load('en_core_web_sm')
import unicodedata
from contractions import contractions_dict

app = Flask(__name__)
model = pickle.load(open("amazon.pkl", "rb"))
cv = pickle.load(open("cv.pkl", "rb"))

# The preprocessing methods are mostly taken from the CS5079 tutorial
def remove_accented_chars(text):
    text1 = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    # if text != text1:
    #     print(text)
    return text1

def remove_special_characters(text):
    text = re.sub('[^a-zA-z0-9\s]', '', text)
    return text


def expand_contractions(text, contraction_mapping=contractions_dict):
    contractions_pattern = re.compile('({})'.format('|'.join(contractions_dict.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) if contraction_mapping.get(
            match) else contraction_mapping.get(match.lower())
        return first_char + expanded_contraction[1:] if expanded_contraction != None else match

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

def lemmatize_text(text):
    text = nlp(text)
    return ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])


# nltk.download('stopwords')
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')


def remove_stopwords(text, is_lower_case=False):
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]

    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def normalize_corpus(corpus, html_stripping=True, contraction_expansion=True,
                     accented_char_removal=True, text_lower_case=True,
                     text_lemmatization=True, special_char_removal=True,
                     stopword_removal=True):
    normalized_corpus = []
    # normalize each document in the corpus
    for doc in corpus:
        # strip HTML
        if html_stripping:
            doc = BeautifulSoup(doc, "lxml").text
        # remove accented characters
        if accented_char_removal:
            doc = remove_accented_chars(doc)
        # expand contractions
        if contraction_expansion:
            doc = expand_contractions(doc)
        # lowercase the text
        if text_lower_case:
            doc = doc.lower()
        # remove extra newlines
        doc = re.sub(r'[\r|\n|\r\n]+', ' ', doc)
        # insert spaces between special characters to isolate them
        special_char_pattern = re.compile(r'([{.(-)!}])')
        doc = special_char_pattern.sub(" \\1 ", doc)
        # lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)
        # remove special characters
        if special_char_removal:
            doc = remove_special_characters(doc)
            # remove extra whitespace
        doc = re.sub(' +', ' ', doc)
        # remove stopwords
        if stopword_removal:
            doc = remove_stopwords(doc, is_lower_case=text_lower_case)

        normalized_corpus.append(doc)
    return normalized_corpus

@app.route('/')
def home():
    return render_template('main.html')


@app.route('/predict', methods=['POST'])  # collect user input
def predict():

    text_input = request.form['review']
    data = [text_input]
    normalized = normalize_corpus(data)
    df = pd.DataFrame(normalized)
    df1 = df.values.flatten()

    cv_test_features = cv.transform(df1)
    prediction = model.predict(cv_test_features)
    return render_template('main.html',
                           prediction_text=(
                               "This review is positive." if (prediction == 1) else "This review is negative."),
                           review=request.form["review"])


if __name__ == '__main__':
    app.run(debug=True)


