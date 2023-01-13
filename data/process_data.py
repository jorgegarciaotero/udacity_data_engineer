import sys
import pandas as pd
import numpy as np
import sqlalchemy
import sqlite3
from sqlalchemy import create_engine

def load_data(
	messages_filepath, categories_filepath):
    '''
    input: filepaths of CSV files
    output: raw merged dataframe
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(categories,messages,how='left',left_on='id',right_on='id')
    return df


def clean_data(
	df):
    '''
    input: raw dataframe
    output: dataframe with categories transformed and converted to binary.
    '''
    categories = df['categories'].str.split(";",expand=True)
    row = categories.iloc[0]
    category_colnames = row
    categories.columns = category_colnames
    categories.columns = categories.columns.str.replace(r'[-10]', '')
    for column in categories:
    	#set each value to be the last character of the string
    	categories[column]=categories[column].str.strip().str[-1].str.replace('2','1')
        #convert column from string to numeric
    	categories[column] = categories[column].astype(float).astype(int)
    categories=categories.rename(columns={'1':'related'})
    df=df.drop('categories',axis=1)
    df = pd.concat([df,categories],axis=1)
    df=df.drop_duplicates('id')
    return df


def save_data(
	df, database_filename):
    '''
    input: dataFrame to be solved into sqlite3 database passed by parameter
    '''
    engine = create_engine('sqlite:///'+database_filename)
    connection = engine.raw_connection()
    df.to_sql('new_table', engine, index=False,if_exists='replace')    
    return

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        print("Cleaned DF: ",df)
 
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
