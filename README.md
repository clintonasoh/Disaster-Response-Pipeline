# Disaster Response Pipeline Project

### Project Summary

In the Project Workspace, you'll find a data set containing real messages that were sent during disaster events. I will be creating a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.

My project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. This project will show off your software skills, including your ability to create basic data pipelines and write clean, organized code!

Below are a few screenshots of the web app.

![disaster-response-project1](disaster-response-project1.png)

![disaster-response-project2](disaster-response-project2.png)


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database

        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves

        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.

    `python run.py`

3. Go to http://0.0.0.0:3001/



ENJOY!
