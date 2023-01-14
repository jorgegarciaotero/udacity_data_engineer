import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Scatter 
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    '''
    input: text column from dataframe to tokenize.
    output: tokens to use the CountVectorized method
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


# load data
engine = create_engine('sqlite://///home/workspace/data/DisasterResponse.db')
df = pd.read_sql_table('new_table', engine)
# load model
model = joblib.load("/home/workspace/models/classifier.pkl")
print(df.dtypes)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    '''
    Route to index page. Graphs are build here.
    '''
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    print("genres: ",genre_names)
    print("genre_counts",genre_counts)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graph1 = {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of categories found per message',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Count"
                }
            }
        }
    
    #Top occurences for every column
    count_ones = lambda x: (x == 1).sum()
    ones = df[['storm','fire','earthquake','water','other_weather','floods'  ]].apply(count_ones).sort_values()
    sorted_ones = ones.sort_values()
    data = [Bar(
        x=sorted_ones.index,
        y=sorted_ones.values
    )]
    layout={
                'title': 'Distribution of messages by natural climate issue',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "climate issue"
                }
	}

    graph2={
            'data':data,
	    'layout':layout
        }

    #scatter
    other_infrastructure=df.groupby('other_infrastructure').count()['message']
    buildings=df.groupby('buildings').count()['message']
    hospitals=df.groupby('hospitals').count()['message']
    aid_centers=df.groupby('aid_centers').count()['message']
    shops=df.groupby('shops').count()['message']
    transport=df.groupby('transport').count()['message']
    shelter=df.groupby('shelter').count()['message']
    my_list=[hospitals[1],other_infrastructure[1],buildings[1],aid_centers[1],shops[1],transport[1],shelter[1]]
    
    bubble_size=list(map(lambda x: x/25, my_list)) 
    
    data=[Scatter(
        x=my_list,#[0,10,20,30,40,50,60,70],
	y=my_list,#[0,10,20,30,40,50,60,70],
	text=['hospitals','other infra','buildings','aid_centers','shops','transport','shelter'],
	marker=dict(
            color=['rgb(93, 164, 214)', 'rgb(255, 144, 14)',  
		  'rgb(44, 160, 101)', 'rgb(255, 65, 54)',
                  'rgb(153, 140, 201)', 'rgb(124, 260, 301)', 
                  'rgb(411, 170, 11)'], 
            size=bubble_size
	)
    )]
   
    layout={
                'title': 'Distribution of messages by climate issue (buildings and vehicles)',
                'yaxis': {
                    'title': "-"
                },
                'xaxis': {
                    'title': "Count"
                }
        }
 
    graph3={
         'data':data,
	 'layout':layout
    } 
    
    graphs=[graph1,graph2,graph3]
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    '''
    Route to text classifier.
    '''
    # save user input in query
    query = request.args.get('query', '') 
    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))
    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
