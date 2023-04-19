# Copyright 2023 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module for feature engineering."""

import os
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import datetime
from datetime import date, datetime, timedelta
from typing import Any, Collection, Dict, List, Mapping, Optional
from google.cloud import bigquery
import calendar
import warnings
warnings.filterwarnings("ignore")

# Define the file paths of the SQL queries for running the preprocessing data pipeline
_DATA_PREPROCESSING_SQL_PATH = 'src/data_processing_SQL_query/BQ_data_preprocessing.sql'
_COMBINE_STAGING_TABLES_SQL_PATH = 'src/data_processing_SQL_query/combine_staging_tables.sql'
_CREATE_FINAL_MODELING_DATA_SQL_PATH = 'src/data_processing_SQL_query/create_final_modeling_datasets.sql'


def read_file(file_name: str) -> str:
    """Reads file."""
    with open(file_name, 'r') as f:
        return f.read()

def overwrite_data_from_dataframe_to_bq(dataset_id: str,
                                        table_name: str,
                                        bigquery_client: bigquery.Client,
                                        df: pd.DataFrame):
    """Load pandas dataframe to Bigquery."""  
    bq_table_id = '{}.{}'.format(dataset_id, table_name)
    job_config = bigquery.job.LoadJobConfig()
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
    job = bigquery_client.load_table_from_dataframe(df, bq_table_id, job_config=job_config)
    job.result()  # Wait for the job to complete.
    table = bigquery_client.get_table(bq_table_id)  # Make an API request.
    print(
        f"Loaded {table.num_rows} rows and {len(table.schema)} columns to {bq_table_id}"
    )

def drop_all_tables_from_given_dataset(bigquery_client: bigquery.Client,
                                       project_id: str,
                                       dataset_id: str,
                                       location: str = 'europe-west4') -> pd.DataFrame:
    """Drop all the tables from the given dataset id.

    Args:
        bigquery_client: BigQuery client.
        project_id: GCP project.
        dataset_id: BigQuery dataset.
        location: Bigquery data location.

    Returns:
        Dataframe with test data.
    """

    select_all_tables_query = "select concat(table_schema,'.',   table_name) AS table_name FROM {}.{}.INFORMATION_SCHEMA.TABLES ORDER BY table_name desc".format(project_id, dataset_id)
    #print(select_all_tables_query)
    table_names_df = bigquery_client.query(select_all_tables_query).result().to_dataframe()

    for table_name in table_names_df.table_name.unique():
        drop_table_query = "DROP TABLE IF EXISTS {};".format(table_name)
        bigquery_client.query(drop_table_query).result()




def create_bucket_features_based_for_numeric_features(data: pd.DataFrame, 
                                                      labels: List[str], 
                                                      bins: List[float],
                                                      numeric_feature_field_name: str,
                                                      bucket_feature_field_name: str) -> pd.DataFrame:
    """Create bucket features based on numeric feature data and remove the numeric feature.

    Args:
        data: Input dataframe.
        labels: Input labels for the bucket feature.
        bins: Input bins for the numeric feature.
        numeric_feature_field_name: Column name of the original numeric feature.
        bucket_feature_field_name: Column name of the new bucket feature.

    Returns:
        New dataframe with the bucket features.
    """     
    
    data[bucket_feature_field_name] = pd.cut(data[numeric_feature_field_name], bins = bins, labels = labels)
    del data[numeric_feature_field_name]
    return data


def further_feature_engineering_for_customer_past_purchase_history(bigquery_client: bigquery.Client,
                                                                   project_id: str,
                                                                   dataset_id: str,
                                                                   location: str = 'europe-west4') -> pd.DataFrame:

    """Create features based on past customer purchase history.

    Args:
        bigquery_client: BigQuery client.
        project_id: GCP project.
        dataset_id: BigQuery dataset.
        location: Bigquery data location.

    Returns:
        Dataframe with past customer purchase history features.
    """
    customer_past_purchase_history_df = bigquery_client.query("SELECT * FROM `{}.{}.customer_past_purchase_history_features_staging_data`".format(project_id, dataset_id)).result().to_dataframe()
    customer_past_purchase_history_df['past_transaction_date'] = pd.to_datetime(customer_past_purchase_history_df['past_transaction_date'])
    customer_past_purchase_history_df['current_transaction_date'] = pd.to_datetime(customer_past_purchase_history_df['current_transaction_date'])
    customer_past_refund_history_df = customer_past_purchase_history_df.loc[customer_past_purchase_history_df['past_transaction_refund_amt']>0]

    # Create transaction level past purchase history features:
    transaction_level_past_transaction_refund_statistics = customer_past_purchase_history_df.groupby(['customer_email_hash', 'transaction_id', 'current_transaction_date'])[['past_transaction_transaction_amt', 'past_transaction_refund_product_quantity', 'past_transaction_refund_amt', 'past_transaction_product_count', 'past_transaction_unique_product_count', 'past_transaction_refund_product_count', 'past_transaction_refund_unique_model_count']].agg(['sum', 'mean', 'median', 'min', 'max', 'std'])
    transaction_level_past_transaction_refund_statistics.columns = ["_".join(x) for x in transaction_level_past_transaction_refund_statistics.columns.ravel()] 
    transaction_level_past_transaction_refund_statistics.reset_index(inplace=True)
    transaction_level_past_transaction_refund_statistics[transaction_level_past_transaction_refund_statistics.select_dtypes(np.number).columns].fillna(0, inplace=True)

    # Create a first_transaction_boolean field: when this is the first transaction (there's no transaction before the given transaction) the boolean is labeled as True
    transaction_level_past_transaction_refund_statistics['first_transaction_boolean']=0
    transaction_level_past_transaction_refund_statistics.loc[transaction_level_past_transaction_refund_statistics['past_transaction_transaction_amt_sum']==0, 'first_transaction_boolean']=1
    # Compute the number of past transactions
    transaction_level_past_transaction_count=customer_past_purchase_history_df.loc[customer_past_purchase_history_df['past_transaction_transaction_amt']>0].groupby('transaction_id').agg(past_transaction_count=('past_transaction_transaction_amt', 'size'))
    transaction_level_past_transaction_count.reset_index(inplace=True)
    # Compute the number of past transactions with refund
    transaction_level_past_refund_transaction_count = customer_past_refund_history_df.groupby('transaction_id').agg(past_refund_count=('past_transaction_transaction_amt', 'size'))
    transaction_level_past_refund_transaction_count.reset_index(inplace=True)
    # Merge the past transaction count data with the past refund data
    transaction_level_past_transaction_refund_count = pd.merge(transaction_level_past_transaction_count, transaction_level_past_refund_transaction_count, how='left', on='transaction_id')
    transaction_level_past_transaction_refund_count.fillna(0, inplace=True)
    # Merge the past transaction & refund count data with the past transaction and refund statistics data:
    transaction_level_customer_past_behaviors_features = pd.merge(transaction_level_past_transaction_refund_statistics, transaction_level_past_transaction_refund_count, on='transaction_id', how='left')

    # compute the most recent refund amount
    most_recent_refund_filter = customer_past_refund_history_df.groupby('transaction_id')['past_transaction_date'].idxmax()
    transaction_level_most_recent_refund_df = customer_past_refund_history_df.loc[most_recent_refund_filter, ['transaction_id','past_transaction_refund_amt']]
    transaction_level_most_recent_refund_df.rename(columns={'past_transaction_refund_amt': 'past_transaction_most_recent_refund_amt'}, inplace=True)
    transaction_level_customer_past_behaviors_features = pd.merge(transaction_level_customer_past_behaviors_features, transaction_level_most_recent_refund_df, on='transaction_id', how='left')

    # compute the most transaction amount
    most_recent_transaction_filter = customer_past_purchase_history_df.loc[customer_past_purchase_history_df['past_transaction_transaction_amt']>0].groupby('transaction_id')['past_transaction_date'].idxmax()
    transaction_level_most_recent_past_transaction_df = customer_past_purchase_history_df.loc[most_recent_transaction_filter, ['transaction_id','past_transaction_transaction_amt']]
    transaction_level_most_recent_past_transaction_df.rename(columns={'past_transaction_transaction_amt': 'past_transaction_most_recent_transaction_amt'}, inplace=True)
    transaction_level_customer_past_behaviors_features = pd.merge(transaction_level_customer_past_behaviors_features, transaction_level_most_recent_past_transaction_df, on='transaction_id', how='left')
    transaction_level_customer_past_behaviors_features[transaction_level_customer_past_behaviors_features.select_dtypes(np.number).columns].fillna(0, inplace=True)

    # compute number of days between most recent refund and current trasanction date
    refund_recency_df = customer_past_refund_history_df.loc[most_recent_refund_filter, ['transaction_id','past_transaction_date', 'current_transaction_date']]
    refund_recency_df['days_between_current_transaction_and_most_recent_refund'] = (refund_recency_df['current_transaction_date'] - refund_recency_df['past_transaction_date']).dt.days
    refund_recency_df['days_between_current_transaction_and_most_recent_refund'] = (refund_recency_df['current_transaction_date'] - refund_recency_df['past_transaction_date']).dt.days
    transaction_level_customer_past_behaviors_features = pd.merge(transaction_level_customer_past_behaviors_features, refund_recency_df[['transaction_id', 'days_between_current_transaction_and_most_recent_refund']], on='transaction_id', how='left')
    transaction_level_customer_past_behaviors_features['days_between_current_transaction_and_most_recent_refund'].fillna(-999, inplace=True)
    # create a bucket/categorical variable for the recency of the refund
    transaction_level_customer_past_behaviors_features = create_bucket_features_based_for_numeric_features(data=transaction_level_customer_past_behaviors_features, 
                                                                                                           labels=["no refund", "within a week", "1 week to 2 weeks", "2 weeks to 1 month", "1 - 3 months", "3 months - 1 year", "more than 1 year"], 
                                                                                                           bins=[-np.inf,0, 7, 14, 30, 90, 365, np.inf],
                                                                                                           numeric_feature_field_name='days_between_current_transaction_and_most_recent_refund',
                                                                                                           bucket_feature_field_name='past_transaction_refund_recency_in_days_bucket')
    # create a bucket field for refund amount percentage at transaction level
    transaction_level_customer_past_behaviors_features['refund_amount_percentage'] =transaction_level_customer_past_behaviors_features['past_transaction_refund_amt_sum']/transaction_level_customer_past_behaviors_features['past_transaction_transaction_amt_sum']
    transaction_level_customer_past_behaviors_features['refund_amount_percentage'].fillna(-999, inplace=True)
    transaction_level_customer_past_behaviors_features = create_bucket_features_based_for_numeric_features(data=transaction_level_customer_past_behaviors_features, 
                                                                                                           labels=["no past transaction", "within 10%", "10%-25%", "25%-50%", "50%-75%", "over 75%"], 
                                                                                                           bins=[-np.inf, 0, 0.1, 0.25, 0.5, 0.75,np.inf],
                                                                                                           numeric_feature_field_name='refund_amount_percentage',
                                                                                                           bucket_feature_field_name='past_transaction_refund_amount_percentage_bucket')
    # create a bucket field for refund count percentage at transaction level
    transaction_level_customer_past_behaviors_features['past_transaction_refund_count_percentage'] = transaction_level_customer_past_behaviors_features['past_refund_count']/transaction_level_customer_past_behaviors_features['past_transaction_count']
    transaction_level_customer_past_behaviors_features['past_transaction_refund_count_percentage'].fillna(-999, inplace=True)
    transaction_level_customer_past_behaviors_features = create_bucket_features_based_for_numeric_features(data=transaction_level_customer_past_behaviors_features, 
                                                                                                           labels=["no past transaction", "within 10%", "10%-25%", "25%-50%", "50%-75%", "over 75%"], 
                                                                                                           bins=[-np.inf, 0, 0.1, 0.25, 0.5, 0.75,np.inf],
                                                                                                           numeric_feature_field_name='past_transaction_refund_count_percentage',
                                                                                                           bucket_feature_field_name='past_transaction_refund_count_percentage_bucket')


    return transaction_level_customer_past_behaviors_features


def further_feature_engineering_for_crm_features(bigquery_client: bigquery.Client,
                                                 transaction_level_customer_past_behaviors_features: pd.DataFrame,
                                                 project_id: str,
                                                 dataset_id: str,
                                                 location: str = 'europe-west4') -> pd.DataFrame:

    """Create features based on crm data and loads the data to Bigquery.

    Args:
        bigquery_client: BigQuery client.
        transaction_level_customer_past_behaviors_features: transaction level customer past behavior data.
        project_id: GCP project.
        dataset_id: BigQuery dataset.
        location: Bigquery data location.

    Returns:
        Dataframe with crm features.
    """
    
    # Create customer level attributes based on the customer crm dataset
    crm_df = bigquery_client.query("SELECT * FROM `{}.{}.crm_preprocessed_table`".format(project_id, dataset_id)).result().to_dataframe()

    # create registered user flag
    crm_df['registered_user_boolean']=1
    crm_df.loc[pd.isnull(crm_df['REGISTRATION']), 'registered_user_boolean']=0

    # merge customer past transaction crm features 
    customer_features_df = pd.merge(transaction_level_customer_past_behaviors_features, crm_df, how='left', on=['customer_email_hash', 'transaction_id'])

    # Create a field to describe customer tenure
    customer_features_df['current_transaction_date'] = pd.to_datetime(customer_features_df['current_transaction_date'])
    customer_features_df['REGISTRATION'] = pd.to_datetime(customer_features_df['REGISTRATION'])
    customer_features_df['user_tenure_in_days'] = customer_features_df['current_transaction_date'] - customer_features_df['REGISTRATION']
    customer_features_df['user_tenure_in_days'] = customer_features_df['user_tenure_in_days'].apply(lambda x: x.days)
    transaction_level_customer_past_behaviors_features = create_bucket_features_based_for_numeric_features(data=customer_features_df, 
                                                                                                           labels=["non registered user", "first time purchaser", "within a week", "1 - 2 weeks", "2 weeks - 1 month", "1 month - 3 months", "3 months - 1 year", "more than 1 year"], 
                                                                                                           bins=[-np.inf, -0.9, 0, 7, 14, 30, 90, 365, np.inf],
                                                                                                           numeric_feature_field_name='user_tenure_in_days',
                                                                                                           bucket_feature_field_name='customer_tenure_bucket')
    del customer_features_df['current_transaction_date']
    #export the data to bigquery table:
    overwrite_data_from_dataframe_to_bq(bigquery_client=bigquery_client, df=customer_features_df, dataset_id=dataset_id, table_name='customer_level_features_data')
    
    return customer_features_df


def further_feature_engineering_for_current_transaction_features(bigquery_client: bigquery.Client,
                                                                 project_id: str,
                                                                 dataset_id: str,
                                                                 location: str = 'europe-west4') -> pd.DataFrame:

    """Create features & attributes on current transaction.

    Args:
        bigquery_client: BigQuery client.
        project_id: GCP project.
        dataset_id: BigQuery dataset.
        location: Bigquery data location.

    Returns:
        Dataframe with current transaction attributes.
    """
    
    transaction_product_level_attributes_df = bigquery_client.query("SELECT * FROM `{}.{}.transaction_product_level_quantity_staging_data`".format(project_id, dataset_id)).result().to_dataframe()
    transaction_level_attributes_df = bigquery_client.query("SELECT * FROM `{}.{}.transaction_level_attributes_staging_data`".format(project_id, dataset_id)).result().to_dataframe()

    # create features on whether the transaction has duplicated products
    transactional_level_product_quanity_df = transaction_product_level_attributes_df.groupby(['transaction_id', 'product_name'])['transaction_total_product_quantity'].sum().reset_index()
    max_transaction_level_same_product_quantity_df = transactional_level_product_quanity_df.groupby('transaction_id')['transaction_total_product_quantity'].max().reset_index()
    max_transaction_level_same_product_quantity_df.rename(columns={'transaction_total_product_quantity': 'transaction_level_max_product_quantity'}, inplace=True)
    max_transaction_level_same_product_quantity_df['transaction_level_duplicate_product_boolean']=(max_transaction_level_same_product_quantity_df['transaction_level_max_product_quantity']>1).astype('int')

    # create features on whether the transaction has different sizes for the same product
    transaction_product_level_unique_product_size_count = transaction_product_level_attributes_df.groupby(['transaction_id', 'product_name'])['product_size'].nunique().reset_index()
    transaction_level_same_product_max_nunique_size_count_df = transaction_product_level_unique_product_size_count.groupby('transaction_id')['product_size'].max().reset_index()
    transaction_level_same_product_max_nunique_size_count_df.rename(columns={'product_size': 'transaction_level_same_product_max_unique_product_size_count'}, inplace=True)
    transaction_level_same_product_max_nunique_size_count_df['transaction_level_same_product_different_size_boolean'] = (transaction_level_same_product_max_nunique_size_count_df['transaction_level_same_product_max_unique_product_size_count']>1).astype('int')
    current_transaction_features_df = pd.merge(pd.merge(transaction_level_attributes_df, max_transaction_level_same_product_quantity_df, on=['transaction_id']), transaction_level_same_product_max_nunique_size_count_df, on=['transaction_id'])

    # create average price for products at transaction level
    transaction_level_avg_price_df = transaction_product_level_attributes_df.groupby('transaction_id')['product_price'].mean().reset_index()
    transaction_level_avg_price_df.rename(columns={'product_price': 'avg_product_price'}, inplace=True)
    current_transaction_features_df = pd.merge(current_transaction_features_df, transaction_level_avg_price_df, on='transaction_id', how='inner')
    del current_transaction_features_df['transaction_date']
    del current_transaction_features_df['transaction_amt']

    # export the data to bigquery table:
    overwrite_data_from_dataframe_to_bq(bigquery_client=bigquery_client, df=current_transaction_features_df, dataset_id=dataset_id, table_name='current_transaction_feature')
    
    return current_transaction_features_df



def further_feature_engineering_on_product_level_features(bigquery_client: bigquery.Client,
                                                          project_id: str,
                                                          dataset_id: str,
                                                          location: str = 'europe-west4') -> pd.DataFrame:
    
    """Create product level past refund features.

    Args:
        bigquery_client: BigQuery client.
        project_id: GCP project.
        dataset_id: BigQuery dataset.
        location: Bigquery data location.

    Returns:
        Dataframe with product features.
    """

    transaction_level_same_product_past_transactions_refund_stats_df = bigquery_client.query("SELECT * FROM `{}.{}.transaction_level_same_product_past_transaction_refund_stats_staging_data`".format(project_id, dataset_id)).result().to_dataframe()
    transacation_level_product_refund_stats_summary = transaction_level_same_product_past_transactions_refund_stats_df.groupby('transaction_id').agg({'product_refund_rate': ['min', 'max', 'mean'], 'estimated_product_refund_amt':['sum']}).reset_index()
    transacation_level_product_refund_stats_summary.columns = ['_'.join(col) for col in transacation_level_product_refund_stats_summary.columns.values]
    transacation_level_product_refund_stats_summary.rename(columns={'transaction_id_':'transaction_id'}, inplace=True)

    # export the data to bigquery table:
    overwrite_data_from_dataframe_to_bq(bigquery_client=bigquery_client, df=transacation_level_product_refund_stats_summary, dataset_id=dataset_id, table_name='transaction_level_past_same_product_refund_stats')
    
    return transacation_level_product_refund_stats_summary



def preprocess_features_for_refund_prediction(bigquery_client: bigquery.Client,
                                                 project_id: str,
                                                 dataset_id: str,
                                                 destination_dataset_id: str,
                                                 source_table_project_id: str,
                                                 source_table_dataset_id: str,
                                                 source_table_web_analytics_table_name: str,
                                                 source_table_refund_data_table_name: str,
                                                 source_table_transaction_id_to_customer_id_mapping_table_name: str,
                                                 source_crm_table_name: str,
                                                 location: str = 'europe-west4') -> (pd.DataFrame, pd.DataFrame):
    """Preprocess features for refund amount prediction ML model.

    Args:
        bigquery_client: BigQuery client.
        project_id: GCP project.
        dataset_id: BigQuery dataset.
        destination_dataset_id: BigQuery dataset for the destination tables.
        source_table_project_id: source table project id.
        source_table_dataset_id: source table dataset id.
        source_table_web_analytics_table_name: web analytics source table name.
        source_table_refund_data_table_name: refund data source table name.
        source_table_transaction_id_to_customer_id_mapping_table_name: transaction id to customer email mapping source table name.
        source_crm_table_name: customer CRM source data table name.
        location: Bigquery data location.

    Returns:
        Data Frames used for prediction and training for both first time transactions and non first time transactions.

    """
    # drop all the preprocessed intermediate tables before kicking off the SQL query
    drop_all_tables_from_given_dataset(bigquery_client, project_id, dataset_id, location)
    # create preprocessed tables: Run all the SQL scripts for data preprocessing and feature engineering
    query = read_file(_DATA_PREPROCESSING_SQL_PATH)
    substituted_query = query.format(project_id = project_id,
                                     dataset_id = dataset_id,
                                     source_table_project_id = source_table_project_id,
                                     source_table_dataset_id = source_table_dataset_id,
                                     source_table_web_analytics_table_name = source_table_web_analytics_table_name,
                                     source_table_refund_data_table_name = source_table_refund_data_table_name, 
                                     source_table_transaction_id_to_customer_id_mapping_table_name = source_table_transaction_id_to_customer_id_mapping_table_name,
                                     source_crm_table_name = source_crm_table_name)

    bigquery_client.query(substituted_query).result()
    
    # further feature engineering through Python script
    transaction_level_customer_past_behaviors_features=further_feature_engineering_for_customer_past_purchase_history(bigquery_client, project_id, dataset_id, location)

    customer_features_df=further_feature_engineering_for_crm_features(bigquery_client, transaction_level_customer_past_behaviors_features, project_id, dataset_id, location)

    current_transaction_features_df=further_feature_engineering_for_current_transaction_features(bigquery_client, project_id, dataset_id, location)

    transacation_level_product_refund_stats_summary=further_feature_engineering_on_product_level_features(bigquery_client, project_id, dataset_id, location)

    # create finalized tables
    query = read_file(_COMBINE_STAGING_TABLES_SQL_PATH)
    substituted_query = query.format(project_id = project_id,
                                     dataset_id = dataset_id)
    bigquery_client.query(substituted_query).result()

    query = read_file(_CREATE_FINAL_MODELING_DATA_SQL_PATH)
    substituted_query = query.format(project_id = project_id,
                                     dataset_id = dataset_id)
    bigquery_client.query(substituted_query).result()

    finalized_modeling_df_for_first_time_transaction = bigquery_client.query("SELECT * FROM `{}.{}.final_modeling_for_firsttime_transaction_df`".format(project_id, dataset_id)).result().to_dataframe()
    finalized_modeling_df_for_non_first_time_transaction = bigquery_client.query("SELECT * FROM `{}.{}.final_modeling_for_non_first_time_transaction_df`".format(project_id, dataset_id)).result().to_dataframe()

    overwrite_data_from_dataframe_to_bq(dataset_id=destination_dataset_id,table_name='final_modeling_for_firsttime_transaction_df',bigquery_client=bigquery_client, df=finalized_modeling_df_for_first_time_transaction)

    overwrite_data_from_dataframe_to_bq(dataset_id=destination_dataset_id,table_name='final_modeling_for_non_first_time_transaction_df',bigquery_client=bigquery_client, df=finalized_modeling_df_for_non_first_time_transaction)
    
    return finalized_modeling_df_for_first_time_transaction, finalized_modeling_df_for_non_first_time_transaction





