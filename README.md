# Disaster Response Pipeline Project of Data Engineering nano degree from [Udacity](https://learn.udacity.com/paid-courses/cd0018/)
---
### Objective: 
The objective of this project is to put in practice all the knowledge aquired during this course, which includes:
- **ETL**: Includes Basic SQLAlchemy usage (using sqlite3) to store and query dataframes and dataframe manipulation and merging.
- **Machine Learning**: Includes using ML Pipelines, sklearn functions for classifying text, TF-IDF, crossvalidating with GridSearchCV,...
- **Presentation in a web server**: Using Python's Flask WS and Matplotlib.

### Files provided:
For suitable develop the project, the following files were provided:
- Excel files: disaster_categories.csv, disaster_messages.csv (/home/workspace/data)
- Python files:
1. process_data.py (/home/workspace/data). Contains the functions used in the ETL Process: Data loading from .CSV, data transformations and data storage in SQLITE3 database. The execution of this scripts stored the merged and transformed dataframes into DisasterResponse.db.
2. train_classifier.py (/home/workspace/models). Contains the Machine Learning pipeline. It reads the DisasterResponse.db file, tokenize the data, build the model, evaluates it and saves it into a .pkl file. 
3. run.py (/home/workspace/app). Executable that launches Flask server in development mode. 




### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.
    - The main directory is: /home/workspace
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
![image](https://user-images.githubusercontent.com/46486273/211402525-f57a42fb-3534-4c97-82da-20e177be5d63.png)

![image](https://user-images.githubusercontent.com/46486273/211410812-1b58f3f4-5c50-46a6-b8d8-5871b91c3bce.png)


2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
