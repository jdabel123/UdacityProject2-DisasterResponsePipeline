import sys
from sqlalchemy import create_engine
import pandas as pd

import re

import nltk

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text  import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score


def load_data(database_filepath):
    '''
    Function to load data from database and to handle categorical variables. Returns the dataframe split into X and y values.
    
    INPUT - database file path expected string.
    
    OUTPUT - returns 
    '''
    
    engine = create_engine(database_filepath)
    
    df = pd.read_sql('Categories', database_filepath)
    
    df = pd.concat([df.drop('genre', axis=1),pd.get_dummies(df['genre'],prefix='Genre',prefix_sep='-',drop_first = True)],axis =1)
    
    X = df['message'].astype(str)

    Y = df.iloc[:,4:].astype('int64')
    
    category_names = Y.columns    
    
    return X, Y, category_names

class StartingVerbTransformer(BaseEstimator,TransformerMixin):
    
    '''
    Class to build a custom transformer to identify whether a message starts with a verb.
    '''
    
    def starting_verb(self,text):
        '''
        Function to determine whether message starts with a verb.
        
        INPUT - text expected list of messages
        
        OUTPUT - returns a 1 dimensional dataframe of boolean answers whether message starts with verb.
        '''
        
        sentence_list = sent_tokenize(text)
        
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            try:
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB','VBP'] or first_word == 'RT':
                    return True
                return False
    
            except:
                return False
            
    
    
    def fit(self,X,y = None):
        return self

        
    def transform(self,X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        
        #Converts boolean answers to binary.
        X_tagged = X_tagged.apply(lambda x: 1 if x == True else 0)
        return pd.DataFrame(X_tagged)

def tokenize(text):
    '''
    Function to tokenize and lemmatize the text
    
    INPUT - array of messages, expected strings
    
    OUTPUT - returns a list of clean tokenized messages
    '''
    
    text = re.sub(r'[^a-zA-Z0-9]',' ', str(text))
       
    tokens = word_tokenize(str(text).lower().strip())
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    
    for token in tokens:
        
        clean_tok = lemmatizer.lemmatize(token)
        
        clean_tokens.append(clean_tok)


    return clean_tokens
    


def build_model():
    '''
    Function to build the machine learning pipeline using a feature union.
    
    INPUT - No expected inputs
    
    OUTPUT - returns model object
    '''
    
    pipeline = Pipeline([
    ('features', FeatureUnion([
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer = tokenize)),
            ('tdidf', TfidfTransformer())    
        ])),
        ('starting_verb', StartingVerbTransformer())   
    ])),
    ('multioutput', MultiOutputClassifier(LinearSVC()))
    ])
    
    params = {'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
          'features__text_pipeline__vect__max_df': (0.75, 1.0)
         }
 
    cv = GridSearchCV(pipeline,param_grid = params,verbose = 3, cv = 3)
     
    return cv
   


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function to print out the classification report for each column.
    
    INPUT - Model object, X_test dataframe, Y_test dataframe and category_names
    
    OUTPUT - Prints classification report for each column.
    """
    y_pred = model.predict(X_test)
    
    for idx, col in enumerate(category_names):
        print(col, classification_report(Y_test.iloc[:,idx], y_pred[:,idx]))
              

def save_model(model, model_filepath):
    '''
    Function to save model to specified path.
    
    INPUT - model object and filepath expected as string.
    
    OUTPUT - Creates and saves the model as a file in specified destintation.
    '''
    
    pickle.dump(model, open(model_filepath,'wb'))

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