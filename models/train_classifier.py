# import libraries
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
import sys
import re
import warnings
warnings.filterwarnings("ignore")

# import NLP libraries
import re
import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet') # download for lemmatization

# import sklearn
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier


def load_data(database_filepath, table_name='messages_categories'):
    """
    Load database and read processed data
    Args:
     database_filepath: filepath of sqlite database
    
    Return:
       X: Features
       Y: Labels
       categores: List of categories columns      
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table(table_name, engine)
    
    X = df.message
    y = df.iloc[:,4:]
    category_names = list(df.columns[4:])
    
    return X, y, category_names


def tokenize(text):
    """
    Tokenizes data, lemmatizes the word to root form and removes stop words
    
    Args:
      text: message
    Return:
    clean_tokens: list of clean tokens
      
    """
    # Define url pattern
    url_re = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Detect and replace urls
    detected_urls = re.findall(url_re, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # tokenize sentences
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    # save cleaned tokens
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
    
    # remove stopwords
    STOPWORDS = list(set(stopwords.words('english')))
    clean_tokens = [token for token in clean_tokens if token not in STOPWORDS]
    
    return clean_tokens


def build_model():
    """
    Returns the pipeline
    """
    # build NLP pipeline - count words, tf-idf and AdaBoostClassifier. 
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC(random_state = 0))))
    ])
    parameters = {
                'clf__estimator__estimator__C': [1, 2, 5]
             }
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=2)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Prints results of multi outpur classifier
    
    Args:
       model: Fitted model with data
       X_test: Test data
       Y_test: Test labels
       category_names: list of category names
    """
    y_pred = model.predict(X_test)
    # print the metrics
    for i, col in enumerate(category_names):
        print('{} category metrics: '.format(col))
        print(classification_report(Y_test.iloc[:,i], y_pred[:,i]))


def save_model(model, model_filepath):
    """
    Dumps the fitted model to pickle files
    
    Args:
      model: Fitted model
      model_filepath: Filepath where model needs to be saved
    """
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