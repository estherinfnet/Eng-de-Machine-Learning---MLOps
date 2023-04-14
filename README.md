# Eng-de-Machine-Learning---MLOps
 
# Eng_ML_MIT
## Final project - Eng Machine Learning 

## Goal 

At that moment, we will validate all our knowledge about Machine Learning Engineering. 
We learn in this course how to develop data collection, create pipelines for processing, adapt models for operation and establish update methods. 
For this, we saw concepts such as AutoML, MLOps, Data Visualization and project structure. The following questions will address these themes.

At that moment, we will validate all our knowledge about Machine Learning Engineering. We learn in this course how to develop data collection, create pipelines for processing, adapt models for operation and establish update methods. For this, we saw concepts such as AutoML, MLOps, Data Visualization and project structure. The following questions will address these themes.

To start this task, download the data which is located at this link: https://www.kaggle.com/c/kobe-bryant-shot-selection/data. This is the data of shots made by NBA star Kobe Bryant during his career.

## What to do 
Image taken from: https://graphics.latimes.com/kobe-every-shot-ever/

Create a solution by following the guidelines:

#### 1. The solution created in this project must be made available in a git repository and made available in a repository server (Github (recommended), Bitbucket or Gitlab). The project must comply with Microsoft's TDSP Framework. All artifacts produced must contain information regarding this project (empty or out-of-context documents will not be accepted). Write the link to your repository.

#### 2. We will develop a shot predictor using two approaches (regression and classification) to predict whether the "Black Mamba" (Kobe's nickname) made or missed the basket.To start development, draw a diagram that demonstrates all the necessary steps in an artificial intelligence project from data acquisition, through model creation, to model operation.'

#### 3. Describe the importance of implementing development and production pipelines in a machine learning solution.

#### 4. How do Streamlit, MLFlow, PyCaret and Scikit-Learn tools help build the pipelines described above? The answer should cover the following aspects:

A. Experiment tracking;

B. Training functions;

C. Model health monitoring;

D. Model update;

E. Provisioning (Deployment).

#### 5. Based on the diagram made in question 2, point out the artifacts that will be created throughout a project. For each artifact, indicate its purpose.

#### 6.Implement the data processing pipeline with mlflow, run with the name "PreparacaoDados":

A. The data must be located in "/Data/kobe_dataset.csv"

B. Note that there is missing data in the database! Rows that have missing data should be disregarded. You will also filter the data where the shot_type value equals 2PT Field Goal. Still, for this exercise only the columns will be considered:

i. lat

ii. lon

iii. minutes_remaining

iv. period

v. playoffs

vi. shot_distance

The shot_made_flag variable will be your target, where 0 indicates that Kobe missed and 1 that the basket was made. The resulting dataset will be stored in the "/Data/processed/data_filtered.parquet" folder. Still on this selection, what is the resulting dimension of the dataset?

C. Separate the data into training (80%) and testing (20%) using a random, stratified choice. Store the resulting datasets in "/Data/operalization/base_{train|test}.parquet . Explain how the choice of training and testing affects the result of the final model. Which strategies help to minimize the effects of data bias.

D. Register parameters (% test) and metrics (size of each base) in MlFlow

#### 7. Implement model training pipeline with Mlflow using name "Training"

A. With the data set aside for training, train a model with sklearn logistic regression using the pyCaret library.

B. Record the "log loss" cost function using the testbase

C. With the data set aside for training, train a sklearn classification model using the pyCaret library. The choice of classification algorithm is free. Justify your choice.

D. Record the cost function "log loss" and F1_score for this new model

#### 8. Register the classification model and make it available through MLFlow via API. Now select the data from the original database where the shot_type is equal to the 3PT Field Goal (it will be a new database) and through the requests library, apply the trained model. Publish a table with the results obtained and indicate the new log loss and f1_score.

A. Does the model adhere to this new basis? Justify.

B. Describe how we can monitor the health of the model in the scenario with and without the availability of the response variable for the model in operation

C. Describe the reactive and predictive retraining strategies for the model in operation.

#### 9. Implement an operation monitoring dashboard using Streamlit.

## Dataset Description
This data contains the location and circumstances of every field goal attempted by Kobe Bryant took during his 20-year career. Your task is to predict whether the basket went in (shot_made_flag).

We have removed 5000 of the shot_made_flags (represented as missing values in the csv file). These are the test set shots for which you must submit a prediction. You are provided a sample submission file with the correct shot_ids needed for a valid prediction.

To avoid leakage, your method should only train on events that occurred prior to the shot for which you are predicting! Since this is a playground competition with public answers, it's up to you to abide by this rule.

The field names are self explanatory and contain the following attributes:

action_type
combined_shot_type
game_event_id
game_id
lat
loc_x
loc_y
lon
minutes_remaining
period
playoffs
season 
seconds_remaining
shot_distance
shot_made_flag (this is what you are predicting)
shot_type
shot_zone_area
shot_zone_basic
shot_zone_range
team_id
team_name
game_date
matchup
opponent
shot_id

