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
        ![image](https://user-images.githubusercontent.com/46486273/211411059-09811609-8864-46a4-826e-8a5af80a9f01.png)

    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
![image](https://user-images.githubusercontent.com/46486273/211402525-f57a42fb-3534-4c97-82da-20e177be5d63.png)

![image](https://user-images.githubusercontent.com/46486273/211410964-bd5e2683-2af0-4923-bff7-4fde182f4b2f.png)


2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage


### Website Preview:
![image](https://user-images.githubusercontent.com/46486273/212464694-46ec53fb-a7c4-4eab-90de-cc0873616c58.png)

![image](https://user-images.githubusercontent.com/46486273/212464687-704fe9fd-9e37-48e7-9f63-fd4b870fe3c2.png)
