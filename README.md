This is not an official Google product.

# PRP - Product Return Predictor

## Overview

### Background:

We need to reduce returns for retailers, to increase sustainability and profit. A product can be returned for various reasons, including quality and size issues. There are also challenges such as “wardrobing” which is where customers purchase an item with the intention of returning it once it has been used for a specific purpose.

### Solution:

To address the return issue for our retail advertisers, the  product return solution leverages machine learning to predict return probability at transaction level. The solution aims to predict product refund amount or likelihood of product return at transaction level.

The solution is to create predictive features around the following dimensions to predict the refund amount for a given transaction using AutoML:

- Target variable: the variable we want to predict here which is the refund amount of the transaction.
- Current Transaction attributes: attributes related to the current transaction (e.g. transaction amount, transaction total product quantity, number of product categories in the current transaction, etc.).
- Web traffic activities: description of web activities of the same session of the transaction that happened.
- Customer past behaviors:  past transaction & refund behaviors that happened before the transaction for the non first-time purchase customer that shopped with the brand.
- Customer generic attributes: general attributes of customers coming from the CRM data.
- Product history: past transaction & refund history for the products that are included in the given transaction.
- Product & Customer past interactions:  Past product & customer interactions in terms of refund and transaction given the customer and product associated with the current transaction.


### Activation:

For the use case, ML models are built to predict the refund amount for each transaction. Given that retailers normally have a N-day refund policy (customers can return their products within N days after their purchase), the model aims to predict the refund amount right after the transaction has been completed. After that, instead of just uploading the transaction amount for offline conversion value import on smart bidding, we can upload the adjusted basket value (transaction amount - predicted refund value) for bidding.



## Quick Start


## Requirements

### Data Requirement for the project

The data required for this solution will include:
- GA data (to capture customer web activity) (essential)
  - Sample fields:
    - Past Sessions/hits/events/time spent aggregated by traffic sources, channels, device type
    - Past time spent/sessions/bounces on certain web pages (e.g. product pages, blog pages, registration page, shopping funnel  etc.)
    - Past interaction with products (e.g. add shopping carts)
- Customer transactions and returns data (SKU level) (essential)
  - Sample fields:
    - Order details (order ID, customer ID, date, etc)
    - Customer details (customer ID, etc)
    - Product details (quantity, value, price, etc)
    - Trasaction & Refund (transaction amount, refund amount, refund product quantity, etc.)
- CRM data (Good to have as it provides richer information about the customers)
  - Sample fields:
    - Customer id (with email address)
    - Demographic info (gender, age, geo, registered user, payment method, loyalty program)


## Outputs

1. Transaction Refund Amount Prediction with transaction id and predicted value for refund amount.

2. Model evaluation stats: RMSE, MAPE, Spearman Rank Correlation, Normalized Gini Coefficient, tier level MAPE, Spearman Rank Correlation, Gini index.


## Solution description

PRP is an advertising solution that predicts product refund based transaction, customer and product data for retailers. It is a Python library that is intended to be run in a Google Cloud Notebook within an advertiser's Google Cloud Platform account. Each step of the PRP modelling and prediction process is carried out programatically from the notebook, including scheduling predictions and model retraining. PRP uses the Vertex SDK for Python to interact with the user's Google Cloud Platform.

The steps in the PRP process are outlined below and are set out in Figure 1:
[TODO] martinacocco to put a diagram here.

- Data cleaning & feature engineering

PRP takes in BigQuery tables (e.g. data from Google Analytics or a CRM transaction dataset) and data preprocessing on BigQuery. It will output intermediate tables during the data preprocessing and create extra features using feature_engineering.py module (under the src folder). At the end of the steps, two tables will be created as training datasets for building the predictive models (one for the first-time transactions and one for the non first-time transactions).
The two tables are by default named as:  finalized_modeling_df_for_first_time_transaction and finalized_modeling_df_for_non_first_time_transaction

[TODO] needs to set a variable to determine when to create training dataset and when to create predict dataset.
For model training, the data consists of a set of features and targets, and a column to indicate which rows are in the model training, test and validation sets. For prediction, the data just consists of a set of features. This step also creates a column (predefined_split_column) which assigns transactions to a training, validation and test set (split with random 15% of users as test, 15% in validation and 70% in training).

- Model Training.
PRP will programmatically train a Vertex AI Tabular AutoML model which will then be visible in the Vertex AI dashboard. The default names of the Vertex Dataset and Model are prp_dataset and prp_model. Once the model is trained, the feature importance will be shown in the Vertex AI model interface. The amount of node hours of training is specified using the budget_milli_node_hours argument (default is 1000 milli hours, which is equivalent to 1 hour). AutoML carries out impressive data prepation before creating the model which means it can ingest features that are:

Plain Text (e.g. customer searches on your website)
Arrays (e.g. product pages visited)
Numerical columns (e.g. add on insurance cost)
Categorical columns (e.g. country)
Model evaluation. PRP will run model evaluation on the test data (a set of transactions that were not included in the model training) and report the metrics. The metrics will be appended to the BigQuery table prp_evaluation (which will be created if it does not exist). See the README section Outputs for which metrics are computed.

- Transaction insights. (TBD)
PRP will connect your trained model to the What IF tool which can help you understand the characteristics of your high return transactions.

- Predictions & Scheduling.
The model will make predictions for all the transactions in the input table. PRP will provide you the option to schedule your predictions using the model on a regular basis using Vertex Pipelines from within the Notebook. See the example in the demo notebook which schedules predictions for 1am everyday. Once the schedule is set up, it will be visible in Vertex Pipelines.

- Monitor feature skew and drift. (TBD)
Vertex AI Model Monitoring supports feature skew and drift detection for categorical and numerical input features.. This can be set up in the Vertex UI from your model's endpoint settings. Use this feature to set up email alerts for when the monitoring detects Prediction Drift Training Prediction Skew, which is a useful prompt to retrain the model using fresh data.

