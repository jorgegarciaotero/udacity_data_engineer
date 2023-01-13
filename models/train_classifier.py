import sys
# import libraries
import pandas as pd
import numpy as np
import re
import sqlalchemy
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle


def load_data(
	database_filepath):
    '''
    input: database filepath and name
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    query="SELECT * from new_table"
    df = pd.read_sql(query,engine)
    X = df.message.values #message column
    Y = df.drop(['message', 'genre', 'id', 'original'], axis = 1)
    category_names=Y.columns
    print("CATEGORIAS: ",category_names)
    return X,Y,category_names


def tokenize(
	text):
    '''
    input: text column from dataframe to tokenize.
    output: tokens to use the CountVectorized method
    '''
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    '''
    Creates the pipeline and builds the model.
    CountVectorizer: Converts a collection of text documents into a vector of term/token counts.
		     Takes as tokenizer the function tokenize()
    TfidTransformer: Scickit Learni's feature extraction module that transforms a count matrix to a normalized term 
		     frequency-inverse document frequency representation.
    MultiOutputClassifier: is a class in scikit-learn's multi-output module that can be used to 
		     train a classifier on multiple outputs. It wraps a single estimator and applies it to multiple outputs. 
    RandomForestClassifier: Ensemble learning method that uses multiple decision trees to make predictions
    '''
    pipeline= Pipeline([
        ('vect' , CountVectorizer(tokenizer=tokenize)),
        ('tfidf' , TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(
		n_estimators= 120,
                max_features='sqrt'
		)
	)
    )
    ])
    parameters = {
        'tfidf__use_idf': (True, False)
        }
    model =  GridSearchCV(pipeline, param_grid=parameters)
    return model


def evaluate_model(
	model, X_test, Y_test, category_names):
    '''
    input: Receives the model and test variables in order to predict the output and print the classification results.
    '''
    y_pred = model.predict(X_test)
    for i,col in enumerate(category_names):
        result=classification_report(Y_test[col], y_pred[:, i],digits=3)
        print("i: ",i)
        print("Column: ", col,":\n ",result)
    return


def save_model(
         model, model_filepath):
    '''
    input: model created in steps before.
    output: pickle file of the model.
    '''
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)
    return


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
