# Disaster Response Pipeline Project
- - - -
### Motivation:
I have applied Data Engineering skills to analyze disaster data from Figure Eight and built ML model that classifies disaster messages.

### File Structure:
* environments.txt - Contains the environments requirement for project
* app folder Contains the following
    * static: Contains the images for project
    * templates: Folder containing
        * master.html: Renders home page
        * go.html: Renders the message classifier
    * run.py: Executes the application
* data folder contains the following
    * disaster_categories.csv: dataset with all the categories
    * disaster_messages.csv: dataset with all the messages
    * process_data.py: Runs ETL pipeline for cleaning the scripts
* models folder contains the following
    * train_classifier.py: Runs ML pipeline for training model and evauluating it

### Instructions:
1. Install requirements using "pip3 install -r requirements.txt"
2. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
