import sys
import pandas as pd

from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Function to load two CSV files and return a merged dataframe on Id.
    
    INPUT - messages_filepath, categories_file expected strings
    
    OUTPUT - merged single dataframe
    '''
    
    messages = pd.read_csv('disaster_messages.csv')
    categories = pd.read_csv('disaster_categories.csv')
                             
    df = pd.merge(messages, categories, on = 'id')                         

    return df

def clean_data(df):
    '''
    Function to clean dataframe.
    
    Input - Dataframe
    
    Output - Cleaned Dataframe
    
    '''
    categories = df.categories.str.split(';',expand=True)
    
    #Select the first row
    row = categories.iloc[0,:]
    
    #Use this row to extract a list of column names
    category_column_names = row.apply(lambda x: x[:-2])
    
    #Replace the names of the columns with new column names.
    categories.columns = category_column_names
    
    #Convert category columns to 0 or 1 integers.
    
    for col in categories:
        
        #set each value to last character in string.
        categories[col] = categories[col].apply(lambda x:x[-1])
        
        #Convery from string to integer
        categories[col] = categories[col].astype('int64')
    
    #Replace categories column with new categories dataframe
    df = pd.concat([df,categories],axis = 1)
    df.drop('categories',axis=1,inplace = True)
    
    #remove duplicates
    df.drop_duplicates(subset=['message'],inplace = True)
    
    #Drop columns orignal and child alone.
    df.drop(['child_alone','original'],axis=1,inplace=True)
    
    #The related column has a max value of 2 - replace this with 1.
    
    df['related'] = df['related'].apply(lambda x: 1 if x == 2 else x)
    
    
    
    return df
       

def save_data(df, database_filename):
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql('Categories', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()