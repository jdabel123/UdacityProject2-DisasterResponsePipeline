# Disaster Response Pipeline Project

#Description:

This project is part of the Data Science Nanodegree program in collaboration with Figure 8. The dataset contains
pre-labelled tweets and messages from real life disasters. The aim of the project is to build a NLP and machine learning
pipeline model to categorize the messages on a real time basis.

The project involves 3 processes:

1 - Processing data from two CSV files to create an ETL pipeline to clean and save the date into an SQLite database.

2- Build a machine learning pipeline to categorize messages in multiple categories.

3- Run a web application to show the results in real time.

#Libraries

You will need to have Python 3.6 installed onto your computer and also have the following libraries installed.

- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/stable/index.html)
- [Numpy](https://numpy.org/)
- [NLTK](https://www.nltk.org/)
- [SQAlchemy](https://www.sqlalchemy.org/)
- [Pickle](https://docs.python.org/3/library/pickle.html)
- [Flask](https://flask.palletsprojects.com/en/1.1.x/)
- [Plotly](https://plotly.com/)


# Installing

You can clone this repository:

	`git clone https://github.com/jdabel123/UdacityProject2-DisasterResponsePipeline`

## Executing Program:
1 - You can run the following commands in the project's directory to set up the database, train model and save the model.

2 - To run ETL pipeline to clean data and store the processed data in the database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response_db.db
3 - To run the ML pipeline that loads data from DB, trains classifier and saves the classifier as a pickle file python models/train_classifier.py data/disaster_response_db.db models/classifier.pkl
4 - Run the following command in the app's directory to run your web app. python run.py

5 - Go to http://0.0.0.0:3001/

# Important Files

**data/process_data.py:** ETL pipeline used to extract data from `disaster_messages.csv` and `disaster_categories.csv` and
						  save this data to a SQLite database `DisasterResponse.db`.
**models/train_classifier.py:** Machine learning pipeline which loads the data, trains the model using NLP pipeline and a custom
								transformer, optimizes the parameters using GridSearchCV and saves the trained model as a pickle 								 file.
**run.py:** Program to launch the flask web app.


# Authors

- John Abel - `johnabel1997@gmail.com`

# Acknowledgements

-`Figure Eight` for providing the labelled dataset to complete this project.



