import sys
import numpy as np
import pandas as pd
import re
import pickle
from sqlalchemy import create_engine

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score, f1_score, classification_report

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def load_data(database_filepath):
    '''
    INPUT - database file path
    OUTPUT - category_names: categories names from the dataframe, X: text messages from the dataframe, Y: column names (categories)
    '''
    #pass
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterDataTable', engine)
    X = df.message
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    '''
    INPUT - text messages
    OUTPUT - cleaned tokenized text messages returned as vectors
    '''
    #pass
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    #splitting text to words
    tokens = word_tokenize(text)
    #back the word to its base form
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_token = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_token)

    return clean_tokens


def build_model():
    '''
    INPUT - No Input
    OUTPUT - ML model after creatign the pipeline
    '''
    #pass
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [50]
        }
            
    model = GridSearchCV(pipeline, parameters)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT - model: the model used (GridSearchCV), X_test: messages test dataset,
            Y_test: categories names test dataset, category_names: categories names
    OUTPUT - No output
    '''
    #pass
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test.iloc[:,1:].values, np.array([x[1:] for x in Y_pred]), 
        target_names=category_names))


def save_model(model, model_filepath):
    '''
    INPUT - model: the model used (GridSearchCV), model_filepath: the model file path
    OUTPUT - No output
    '''
    #pass
    pickle.dump(model, open(model_filepath, 'wb'))


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