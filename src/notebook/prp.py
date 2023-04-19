# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main module to train and predict a PRP (product return) model.

The PRP model uses Vertex AI AutoML Tables.


Minimal example:

from google.cloud import bigquery
from sklearn import ensemble
import prp

# Create BigQuery client for cloud authentication.
bigquery_client = bigquery.Client()

# Initiate the PRP class.
# The Google Cloud Platform project will be identified using Bigquery client.
# Initiate the CrystalValue class with the relevant parameters.
pipeline = prp.ProductReturnPredictor(
    project_id = 'project_id',
    dataset_id = 'dataset_id',
    source_table_project_id = 'project_id',
    source_table_dataset_id = 'dataset_id',
    destination_dataset_id = 'dataset_id',
    source_table_transaction_id_to_customer_id_mapping_table_name = 'table_name',
    source_table_web_analytics_table_name = 'table_name',
    source_table_refund_data_table_name = 'table_name',
    source_crm_table_name = 'table_name',
    location = 'europe-west1',
    write_parameters = True,
    parameters_filename = 'prp_model_parameters.json',
    credentials = None,
)

# Perform feature engineering.
# PRP will create features based on the given key source tables including: refund data, web analytics data, crm table and transaction id to customer email mapping data.
# The output will be two dataframes: training data for first-time transactions and training data for non first-time transactions.
finalized_modeling_df_for_first_time_transaction, finalized_modeling_df_for_non_first_time_transaction = pipeline.feature_engineer()

# Model training.
# Creates AI Platform Dataset and trains AutoML model in your GCP.
# You need to pick which dataset you want to training the model by specifying the training_table_name here.
model_object = pipeline.train_automl_model(training_table_name='final_modeling_for_non_first_time_transaction_df')

# You can view your model training progress here:
# https://console.cloud.google.com/vertex-ai/training/training-pipelines
# Once the training is finished, check out your trained AutoML model in the UI.
# Feature importance graphs and statistics on the data can be viewed here.
# https://console.cloud.google.com//vertex-ai/models

# You can also deploy your model to create fast predictions and to create
# product return predictive model evaluation statistics.
pipeline.deploy_model()
# For model evaluation, you need to specify training dataset as well.
model_summary_statistics, tier_wise_model_performance_summary = pipeline.evaluate_model(training_table_name='final_modeling_for_non_first_time_transaction_df')

# Now create product return amount predictions using the model and input data.[TBD]: This part has not been finalized
pipeline.batch_predict(
    input_table_name='predict_features_data',
    destination_table='predictions')

"""

import dataclasses
import json
from typing import Any, Collection, Dict, List, Mapping, Optional

from absl import logging
from google.cloud import bigquery
from google.cloud import storage
from google.cloud import aiplatform
from google.cloud.exceptions import NotFound
import numpy as np
import pandas as pd

from src import automl
from src import feature_engineering
from src import model_evaluation


def load_parameters_from_file(
    filename: str = 'prp_model_parameters.json') -> Dict[str, str]:
  """Reads parameters from local file."""
  logging.info('Reading parameters from file %r', filename)
  with open(filename) as f:
    return json.load(f)


@dataclasses.dataclass
class ProductReturnPredictor:
  """Class to train and predict LTV model.

  Attributes:
    project_id: The Bigquery project id.
    dataset_id: The Bigquery dataset id.
    credentials: The (optional) credentials to authenticate your Bigquery and
      AIplatform clients. If not passed, falls back to the default inferred
      from the environment.
    source_table_project_id: project id of source tables.
    source_table_dataset_id: dataset id of source tables.
    destination_dataset_id: dataset id of destination tables.
    source_table_transaction_id_to_customer_id_mapping_table_name: table name of transaction id to customer email mapping.
    source_table_web_analytics_table_name: table name of web analytics data.
    source_table_refund_data_table_name: table name of refund data.
    source_crm_table_name: table name of crm data.
    training_table_name: table name of training dataset.
    predict_table_name: table name of prediction data.
    location: The Bigquery and Vertex AI location for processing (e.g.
      'europe-west4' or 'us-east-4')
    model_id: The ID of the model that will be created.
    endpoint_id: The ID of the endpoint that will be created for a deployed
      model.
    write_parameters: Whether to write input parameter to file.
    parameters_filename: The file path to write PRP model parameters to.
    bigquery_client: Bigquery client for querying Bigquery.
  """
  project_id: str
  dataset_id: str
  source_table_project_id: str
  source_table_dataset_id: str
  destination_dataset_id: str
  source_table_transaction_id_to_customer_id_mapping_table_name: str
  source_table_web_analytics_table_name: str
  source_table_refund_data_table_name: str
  source_crm_table_name: str
  credentials: Optional[Any] = None
  location: str = 'europe-west4'
  model_id: Optional[str] = None
  endpoint_id: Optional[str] = None
  write_parameters: bool = False
  parameters_filename: str = 'prp_model_parameters.json'
  bigquery_client: Optional[bigquery.Client] = None


  def __post_init__(self):
    logging.info('Using Google Cloud Project: %r', self.project_id)
    logging.info('Using dataset id: %r', self.dataset_id)
    logging.info('source table project id: %r', self.source_table_project_id)
    logging.info('source table dataset id: %r', self.source_table_dataset_id)
    logging.info('source data transaction id to customer id mapping table name: %r', self.source_table_transaction_id_to_customer_id_mapping_table_name)
    logging.info('source data web analytics table name: %r', self.source_table_web_analytics_table_name)
    logging.info('source data crm table name: %r', self.source_crm_table_name)
    #logging.info('training data table name: %r', self.training_table_name)
    #logging.info('prediction table name: %r', self.predict_table_name)
    logging.info('Using Google Cloud location: %r', self.location)

    self.bigquery_client = bigquery.Client(
        project=self.project_id,
        credentials=self.credentials)
    try:
      self.bigquery_client.get_dataset(self.dataset_id)
    except NotFound:
      logging.info('Dataset %r not found, creating the dataset %r '
                   'in project %r in location %r',
                   self.dataset_id, self.dataset_id, self.project_id,
                   self.location)
      self.bigquery_client.create_dataset(self.dataset_id)
    if self.write_parameters:
      self._write_parameters_to_file()

  def _write_parameters_to_file(self) -> None:
    """Writes parameters to file."""
    parameters = {
        'project_id': self.project_id,
        'dataset_id': self.dataset_id,
        'source_table_project_id': self.source_table_project_id,
        'source_table_dataset_id': self.source_table_dataset_id,
        'source_table_transaction_id_to_customer_id_mapping_table_name': self.source_table_transaction_id_to_customer_id_mapping_table_name,
        'source_table_web_analytics_table_name': self.source_table_web_analytics_table_name,
        'source_crm_table_name': self.source_crm_table_name,
        'location': self.location,
        'model_id': self.model_id
    }
    with open(self.parameters_filename, 'w') as f:
      json.dump(parameters, f)
    logging.info('Parameters writen to file: %r',
                 self.parameters_filename)



  def feature_engineer(self) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Builds training & prediction dataset for predict transaction level refund amount.

    This function takes in source tables location (with predefined data structure), project id and dataset id and creates dataset that could be used for model training and prediction.

    Returns:
      Data Frames to use for model training for first transactions and non-first transactions.
    """

    finalized_modeling_df_for_first_time_transaction, finalized_modeling_df_for_non_first_time_transaction = feature_engineering.preprocess_features_for_refund_prediction(bigquery_client = self.bigquery_client,
                                                          project_id = self.project_id,
                                                          dataset_id = self.dataset_id,
                                                          destination_dataset_id = self.destination_dataset_id,
                                                          source_table_project_id = self.source_table_project_id,
                                                          source_table_dataset_id = self.source_table_dataset_id,
                                                          source_table_web_analytics_table_name = self.source_table_web_analytics_table_name,
                                                          source_table_refund_data_table_name = self.source_table_refund_data_table_name,
                                                          source_table_transaction_id_to_customer_id_mapping_table_name = self.source_table_transaction_id_to_customer_id_mapping_table_name,
                                                          source_crm_table_name = self.source_crm_table_name,
                                                          location =  self.location)

    return finalized_modeling_df_for_first_time_transaction, finalized_modeling_df_for_non_first_time_transaction



  def train_automl_model(
      self,
      training_table_name: str,
      dataset_display_name: str = 'prp_dataset',
      model_display_name: str = 'prp_model',
      predefined_split_column_name: str = 'predefined_split_column',
      target_column: str = 'refund_amt',
      optimization_objective: str = 'minimize-rmse',
      budget_milli_node_hours: int = 1000) -> aiplatform.models.Model:
    """Creates Vertex AI Dataset and trains an AutoML Tabular model.

    Args:
      training_table_name: table name of the training data.
      dataset_display_name: The display name of the Dataset to create.
      model_display_name: The display name of the Model to create.
      predefined_split_column_name: A name of one of the Dataset's columns. The
        values of the column must be one of {``training``, ``validation``,
        ``test``}, and it defines to which set the given piece of data is
        assigned. If for a piece of data the key is not present or has an
        invalid value, that piece is ignored by the pipeline.
      target_column: The target to predict.
      optimization_objective: Objective function the Model is to be optimized
        towards. The training task creates a Model that maximizes/minimizes the
        value of the objective function over the validation set. "minimize-rmse"
        (default) - Minimize root-mean-squared error (RMSE). "minimize-mae" -
        Minimize mean-absolute error (MAE). "minimize-rmsle" - Minimize
        root-mean-squared log error (RMSLE). Only used for AutoML.
      budget_milli_node_hours: The number of node hours to use to train the
        model (times 1000), 1000 milli node hours is 1 mode hour. Only used for
        AutoML.

    Returns:
      Vertex AI AutoML model.
    """
    model = self.run_automl_training(
        training_table_name = training_table_name,
        dataset_display_name=dataset_display_name,
        model_display_name=model_display_name,
        predefined_split_column_name=predefined_split_column_name,
        target_column=target_column,
        optimization_objective=optimization_objective,
        budget_milli_node_hours=budget_milli_node_hours)

    self.model_id = model.name
    return model


  def run_automl_training(
      self,
      training_table_name: str,
      dataset_display_name: str = 'prp_dataset',
      model_display_name: str = 'prp_model',
      predefined_split_column_name: str = 'predefined_split_column',
      target_column: str = 'refund_amt',
      optimization_objective: str = 'minimize-rmse',
      budget_milli_node_hours: int = 1000) -> aiplatform.Model:
    """Creates Vertex AI Dataset and trains an AutoML Tabular model.

    An AutoML Dataset is required before training a model. See
    https://cloud.google.com/vertex-ai/docs/datasets/create-dataset-api
    https://cloud.google.com/vertex-ai/docs/training/automl-api

    Args:
      training_table_name: table name of the training data.
      dataset_display_name: The display name of the Dataset to create.
      model_display_name: The display name of the Model to create.
      predefined_split_column_name: A name of one of the Dataset's columns. The
        values of the column must be one of {``training``, ``validation``,
        ``test``}, and it defines to which set the given piece of data is
        assigned. If for a piece of data the key is not present or has an
        invalid value, that piece is ignored by the pipeline.
      target_column: The target to predict.
      optimization_objective: Objective function the Model is to be optimized
        towards. The training task creates a Model that maximizes/minimizes the
        value of the objective function over the validation set. "minimize-rmse"
        (default) - Minimize root-mean-squared error (RMSE). "minimize-mae" -
        Minimize mean-absolute error (MAE). "minimize-rmsle" - Minimize
        root-mean-squared log error (RMSLE).
      budget_milli_node_hours: The number of node hours to use to train the
        model (times 1000), 1000 milli node hours is 1 mode hour.

    Returns:
      Vertex AI AutoML model.
    """

    aiplatform_dataset = automl.create_automl_dataset(
        project_id=self.project_id,
        dataset_id=self.destination_dataset_id,
        table_name= training_table_name,
        dataset_display_name=dataset_display_name,
        location=self.location)

    model = automl.train_automl_model(
        project_id=self.project_id,
        aiplatform_dataset=aiplatform_dataset,
        model_display_name=model_display_name,
        predefined_split_column_name=predefined_split_column_name,
        target_column=target_column,
        optimization_objective=optimization_objective,
        budget_milli_node_hours=budget_milli_node_hours,
        location=self.location)
    self.model_id = model.name
    self._write_parameters_to_file()
    return model


  def batch_predict(self,
                    input_table_name: str,
                    model_id: Optional[str] = None,
                    model_name: str = 'prp_model',
                    destination_table: str = 'prp_predictions'):
    """Creates predictions using Vertex AI model into destination table.

    Args:
      input_table_name: The table containing features to predict with.
      model_id: The resource name of the Vertex AI model e.g.
        '553728129496821'
      model_name: The name of the Vertex AI trained model e.g.
        'prp_model'.
      destination_table: The table to either create (if it doesn't exist) or
        append predictions to within your dataset.
    """
    if not model_id:
      model_id = self.model_id
    batch_predictions = automl.create_batch_predictions(
        project_id=self.project_id,
        dataset_id=self.destination_dataset_id,
        model_id=model_id,
        table_name=input_table_name,
        location=self.location)

    automl.load_predictions_to_table(
        bigquery_client=self.bigquery_client,
        dataset_id=self.destination_dataset_id,
        batch_predictions=batch_predictions,
        location=self.location,
        destination_table=destination_table,
        model_name=model_name)

  def predict(self,
              input_table: pd.DataFrame,
              model_id: Optional[str] = None,
              endpoint_id: Optional[str] = None,
              destination_table: str = 'prp_predictions',
              round_decimal_places: int = 2) -> pd.DataFrame:
    """Creates predictions using Vertex AI model into destination table.

    Args:
      input_table: The table containing features to predict with.
      model_id: The resource name of the Vertex AI model e.g.
        '553728129496821'.
      endpoint_id: The endpoint ID of the model. If not specified, it will be
        found using the model_id.
      destination_table: The table to either create (if it doesn't exist) or
        append predictions to within your dataset.
      round_decimal_places: How many decimal places to round to.

    Returns:
      Predictions.
    """
    if not model_id:
      if not self.model_id:
        raise ValueError('model_id is required for prediction.')
      model_id = self.model_id
    if not endpoint_id:
      if not self.endpoint_id:
        model = aiplatform.Model(model_id, location=self.location)
        endpoint_id = model.gca_resource.deployed_models[0].endpoint.split(
            '/')[-1]
      else:
        endpoint_id = self.endpoint_id

    input_table = input_table.copy()
    input_table['predicted_value'] = np.round(
        automl.predict_using_deployed_model(
            project_id=self.project_id,
            endpoint=endpoint_id,
            features=input_table,
            location=self.location),
        round_decimal_places)

    output = input_table[[
        'transaction_id',
        'predicted_value']]

    table_id = f'{self.project_id}.{self.destination_dataset_id}.{destination_table}'
    try:
      self.bigquery_client.get_table(table_id)
      job_config = bigquery.job.LoadJobConfig(
          write_disposition=bigquery.WriteDisposition.WRITE_APPEND)
      logging.info('Appending to table %r in location %r', table_id,
                   self.location)
    except NotFound:
      job_config = bigquery.job.LoadJobConfig(
          write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE)
      logging.info('Creating table %r in location %r', table_id, self.location)

    self.bigquery_client.load_table_from_dataframe(
        dataframe=output,
        destination=table_id,
        job_config=job_config,
        location=self.location).result()
    return output

  def deploy_model(self, model_id: Optional[str] = None) -> aiplatform.Model:
    """Creates an endpoint and deploys Vertex AI Tabular AutoML model.

    Args:
      model_id: The ID of the model.

    Returns:
      AI Platform model object.
    """
    if not model_id:
      model_id = self.model_id
    model = automl.deploy_model(
        bigquery_client=self.bigquery_client,
        model_id=model_id,
        location=self.location)
    model.wait()
    return model

  def evaluate_model(self,
                     training_table_name,
                     model_id: Optional[str] = None,
                     endpoint_id: Optional[str] = None,
                     table_evaluation_stats: str = 'prp_evaluation',
                     number_bins: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Creates a plot and Big Query table with evaluation metrics for LTV model.

    Args:
     training_table_name: table name of the training dataset.
      model_id: The resource name of the Vertex AI model e.g.
        '553728129496821'.
      endpoint_id: The endpoint ID of the model. If not specified, it will be
        found using the model_id.
      table_evaluation_stats: Destination BigQuery Table to store model results.
      number_bins: Number of bins to split the LTV predictions into for
        evaluation. The default split is into deciles.

    Returns:
      Model evaluation metrics on the test set.

    Raises:
      ValueError if no model_id is specified.
    """
    if not model_id:
      if not self.model_id:
        raise ValueError('model_id is required for prediction.')
      model_id = self.model_id
    if not endpoint_id:
      if not self.endpoint_id:
        model = aiplatform.Model(model_id, location=self.location)
        endpoint_id = model.gca_resource.deployed_models[0].endpoint.split(
            '/')[-1]
      else:
        endpoint_id = self.endpoint_id
    print('endpoint', endpoint_id)

    # generate prediction data
    prediction_df = model_evaluation.generate_model_predictions(bigquery_client=self.bigquery_client,
                                              dataset_id=self.destination_dataset_id,
                                              endpoint=endpoint_id,
                                              model_id=model_id,
                                              training_data_name=training_table_name,
                                              location=self.location)

    model_summary_statistics, tier_wise_model_performance_summary=model_evaluation.evaluate_model_predictions(bigquery_client=self.bigquery_client,
                                                                                                              dataset_id=self.destination_dataset_id,
                                                                                                              model_id=model_id, data=prediction_df,
                                                                                                                        table_evaluation_stats=table_evaluation_stats,
                                                                                                              location=self.location,
                                                                                                              number_bins=number_bins)
    return model_summary_statistics, tier_wise_model_performance_summary

