import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads messages and categories from dataset
    
    Args:
    messages_filepath : filepath of messages.csv
    categories_filepath : filepath of categories.csv
    
    Return:
    df : merged dataframe of messages.csv and categories.csv
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return pd.merge(messages, categories, on = 'id')
    


def clean_data(df):
    """
    Perform below steps:
    1. Cleaning categories
    2. Drop duplicate values
    
    Args:
    df: Dataframe
    
    Return:
    df: Cleaned dataframe
    
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[1]
    category_colnames = row.apply(lambda x: x.split('-')[0])
    # rename the columns of `categories`
    categories.columns = category_colnames
    categories.related.loc[categories.related == 'related-2'] = 'related-1'
    
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: int(x.split('-')[1]))
        

    # drop the original categories column from `df` and join categories
    df.drop(labels='categories', axis=1, inplace=True)
    df = df.join(categories)
    
    # drop duplicates
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename, table_name='messages_categories'):
    """
    save cleaned dataset in to database table
    
    Args:
    df: cleaned dataframe
    database_filename : file path to save database
    table_name: table_name in database to store cleaned data
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql(table_name, engine, index=False)
      


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