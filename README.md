# AutoML_Tables
POC of GCP AutoML Tables

## Objective
The AutoML Tables in Google Cloud AI Platform is an intuitive and ease-to-use supervised learning service, to build and deploy state of the art machine learning model with structured (tabular) data.

The objective of this repo is to validate the capabilities of AutoML Tables for all business users.

We will use Chicago Taxi Trip dataset, to predict the Chicago taxi fare in AutoML Tables. We will focus on testing the following capabilities: easy-to-build models, easy-to-deploy and scale models, and flexible user options.

## The Data 
Source: (https://www.kaggle.com/chicago/chicago-taxi-trips-bq)

### Preparing the Data
1M rows were extracted from BigQuery(upload from local) using SQL. After proper feature engineering, finally got a training dataset(58321 rows) and a test dataset(100 rows).

### Training & Test Datasets
After data preparation, we used 7 features to predict the fare price (in dollars). 

Predictor Features:

* pickup_latitude
* pickup_longitude
* dropoff_latitude
* dropoff_longitude
* weekday (day of the week)
* is_luxury(if the company tend to provide premium services)
* k2(cluster of the driver based on the clustering results)


#### Training Data
The training dataset imported from Google BigQuery consists of 58321 rows by 8 columns (7 predictors + 1 target).
Per AutoML Tables beginner’s guide, by default, the training data we imported to AutoML Tables will be split into 80% training, 10% validation, and 10% test sets, and  we can manually edit those values if necessary. 
(Source: https://cloud.google.com/automl-tables/docs/beginners-guide)


#### Test Data

100 rows by 8 columns, and some rows contain null(for evaluation purpose). 
 
## The Model
The .ipynb file contains 7 sessions

### session 1: The data--gathering and preparing
1. extracted data from GCP BigQuery as 2 Pandas dataframe, named train and test
2. general data analysis lookup
3. ploted distribution graph to check if test dataset is consitent with train dataset in some features

### session 2: Importing training set to AutoML Tables
1. created a service account key in GCP console, saved the key in JSON file. Reference: https://cloud.google.com/iam/docs/creating-managing-service-account-keys
2. created an AutoML client with a GCP project id and compute engine region
3. created and imported train dataset to GCP AutoML Tables
```css
# create dataset in AutoML

dataset_display_name = 'demo3_chicago_taxi_fare' 
create_dataset_response = client.create_dataset(
    location_path,
    {'display_name': dataset_display_name, 'tables_dataset_metadata': {}})
dataset_name = create_dataset_response.name
create_dataset_response
```
```css
# get training data path

dataset_bq_input_uri = '$[bigquery dir here], removed by security reason'

# Define input configuration.
input_config = {
    'bigquery_source': {
        'input_uri': dataset_bq_input_uri
    }
}
```
```css
# Import training data table into AutoML dataset 

import_data_response = client.import_data(dataset_name, input_config)

# Wait until import is done.
import_data_result = import_data_response.result()
import_data_result
```
4. reviewed schema and change it if necessary, declare target and features
```css
# Schema review

import google.cloud.automl_v1beta1.proto.data_types_pb2 as data_types

# List table specs
list_table_specs_response = client.list_table_specs(dataset_name)
table_specs = [s for s in list_table_specs_response]

# List column specs
table_spec_name = table_specs[0].name
list_column_specs_response = client.list_column_specs(table_spec_name)
column_specs = {s.display_name: s for s in list_column_specs_response}
[(x, data_types.TypeCode.Name(
  column_specs[x].data_type.type_code)) for x in column_specs.keys()]
  ```
  ```css
  # Update dataset - split features and target

label_column_name = 'fare_dollars' 
label_column_spec = column_specs[label_column_name]
label_column_id = label_column_spec.name.rsplit('/', 1)[-1]
print('Label column ID: {}'.format(label_column_id))

# Define the values of the fields to be updated.
update_dataset_dict = {
    'name': dataset_name,
    'tables_dataset_metadata': {
        'target_column_spec_id': label_column_id
    }
}

update_dataset_response = client.update_dataset(update_dataset_dict)
update_dataset_response
  ```

### session 3:  Managing and training AutoML Tables model
created a model training in AutoML Tables, and set training budget no longer than 5 hours
```css
model_display_name = 'demo3_model' # give model a random name

# the time budge is 5 hours
model_dict = {
    'display_name': model_display_name,
    'dataset_id': dataset_name.rsplit('/', 1)[-1],
    'tables_model_metadata': {'train_budget_milli_node_hours': 5000}
}

create_model_response = client.create_model(location_path, model_dict)
print('Dataset import operation: {}'.format(create_model_response.operation))
```

### session 4: training performance
printed out general metrics of the AutoML Tables model, such as MSE, MAE, MAPE, R_sqaured, feature importance
```css
# general regression metric results

metrics= [x for x in client.list_model_evaluations(model_name)][-1]
metrics.regression_evaluation_metrics
```
```css
# raw feature importance

model = client.get_model(model_name)
feat_list = [(x.feature_importance, x.column_display_name) for x in model.tables_model_metadata.tables_model_column_info]
feat_list.sort(reverse=True)
feat_list[:15]
```

### session 5: making batch prediction
1. used the test set already uploaded to BigQuery
2. declared path to store predictions(either Cloud Storage or BigQuery)
```css # make prediction and store result in BigQuery 

input_path = '$[bigquery dir here], removed by security reason'
output_path = '$[bigquery dir here], removed by security reason'

from google.cloud import automl_v1beta1 as automl
import csv

automl_client = automl.AutoMlClient()

# Get the full path of the model.
model_full_id = automl_client.model_path(
    project_id, location, model_id
)

# Create client for prediction service.
prediction_client = automl.PredictionServiceClient()

if input_path.startswith('bq'):
    input_config = {"bigquery_source": {"input_uri": input_path}}
else:
    # Get the multiple Google Cloud Storage URIs.
    input_uris = input_path.split(",").strip()
    input_config = {"gcs_source": {"input_uris": input_uris}}

if output_path.startswith('bq'):
    output_config = {"bigquery_destination": {"output_uri": output_path}}
else:
    # Get the multiple Google Cloud Storage URIs.
    output_config = {"gcs_destination": {"output_uri_prefix": output_path}}

# Query model
response = prediction_client.batch_predict(
    model_full_id, input_config, output_config)

print("Making batch prediction... ")
try:
    result = response.result()
except:
    pass
print("Batch prediction complete.\n{}".format(response.metadata))
```

### session 6: evaluating predictions
1. the prediction result is store in nested table (need to unnest the result in Pandas)
this is the result originally in BigQuery: 
```css 
[{'tables': {'prediction_interval': {'end': 23.365476608276367,'start': 5.482821464538574},'value': 9.624988555908203}}]
```
the 'value' is the prediction that we are looking for
2. compared the prediction with the actual fare price, with the metrics such as MSE, MAE, MAPE, R_sqaured
3. plot the predictions and actual values in a graph to compare visually
![alt text](https://github.com/pandakayak/AutoML_Tables/blob/master/plot1.PNG)

### session 7: making online prediction
1. firstly, we need to deploy the model in GCP AutoML Tables, to enable online prediction
```css
! curl -X POST \
-H "Authorization: Bearer $(gcloud auth application-default print-access-token)" \
https://automl.googleapis.com/v1beta1/projects/hackathon1-183523/locations/us-central1/models/TBL4988132411498823680:deploy
```
2. created a sample to make prediction, in JSON file
3. enable online prediction with GCP command
```css 
! curl -X POST -H "Content-Type: application/json" \
  -H "Authorization: Bearer $(gcloud auth application-default print-access-token)" \
  https://automl.googleapis.com/v1beta1/projects/hackathon1-183523/locations/us-central1/models/TBL4988132411498823680:predict \
  -d @request.json
  ```
4. undelopyed the model
```css 
! curl -X POST \
-H "Authorization: Bearer $(gcloud auth application-default print-access-token)" \
https://automl.googleapis.com/v1beta1/projects/hackathon1-183523/locations/us-central1/models/TBL4988132411498823680:undeploy
```
## reference:
1. https://cloud.google.com/automl-tables/docs/
2. https://github.com/GoogleCloudPlatform/python-docs-samples/tree/master/tables/automl/notebooks
