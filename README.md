# Disaster Response Pipeline Project of Data Engineering nano degree from [Udacity](https://learn.udacity.com/paid-courses/cd0018/)
---
### Objective: 
The objective of this project is to put in practice all the knowledge aquired during this course, which includes:
- **ETL**: Includes Basic SQLAlchemy usage (using sqlite3) to store and query dataframes and dataframe manipulation and merging.
- **Machine Learning**: Includes using ML Pipelines, sklearn functions for classifying text, TF-IDF, crossvalidating with GridSearchCV,...
- **Presentation in a web server**: Using Python's Flask WS and Matplotlib.


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
