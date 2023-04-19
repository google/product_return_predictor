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

-- Build predict dataset for a machine learning model for predict transation refund amount.
-- This script is used to merge all features from different data sources target variables, web activities, crm features, current transaction attributes, past same product refund stats and also past same customer same product refund stats.


CREATE OR REPLACE TABLE
  `{project_id}.{dataset_id}.final_modeling_df` AS
  
SELECT 
-- target variables
CAST(target_variable.transaction_id AS STRING) AS transaction_id,
CAST(target_variable.transaction_date AS DATE) AS transaction_date,
CAST(crm_features.customer_email_hash AS STRING) AS customer_email_hash,
IFNULL(CAST(target_variable.transaction_amt AS FLOAT64),0) AS transaction_amt, 
IFNULL(CAST(target_variable.refund_amt AS FLOAT64),0) AS refund_amt,

-- current transaction attributes
IFNULL(CAST(current_transaction_attributes.discount_type AS STRING), 'UNKNOWN') AS discount_type,
IFNULL(CAST(current_transaction_attributes.transaction_delivery_type AS STRING), 'UNKNOWN') AS transaction_delivery_type,
IFNULL(CAST(current_transaction_attributes.transaction_payment_type AS STRING), 'UNKNOWN') AS transaction_payment_type,
IFNULL(CAST(current_transaction_attributes.transaction_total_product_quantity AS FLOAT64),0) AS transaction_total_product_quantity,
IFNULL(CAST(current_transaction_attributes.transaction_shipping_cost AS FLOAT64), 0) AS transaction_shipping_cost,
IFNULL(CAST(current_transaction_attributes.freeshipping_boolean AS FLOAT64), 0) AS freeshipping_boolean,
IFNULL(CAST(current_transaction_attributes.nunique_product_categories AS FLOAT64),0) AS nunique_product_categories,
IFNULL(CAST(current_transaction_attributes.nunique_product_names AS FLOAT64),0) AS nunique_product_names,
IFNULL(CAST(current_transaction_attributes.transaction_level_max_product_quantity AS FLOAT64),0) AS transaction_level_max_product_quantity,
IFNULL(CAST(current_transaction_attributes.transaction_level_duplicate_product_boolean AS FLOAT64),0) AS transaction_level_duplicate_product_boolean,
IFNULL(CAST(current_transaction_attributes.transaction_level_same_product_max_unique_product_size_count AS FLOAT64),0) AS transaction_level_same_product_max_unique_product_size_count,
IFNULL(CAST(current_transaction_attributes.transaction_level_same_product_different_size_boolean AS FLOAT64),0) AS transaction_level_same_product_different_size_boolean,
IFNULL(CAST(current_transaction_attributes.avg_product_price AS FLOAT64),0) AS avg_product_price,

-- web activities before & during the transaction
IFNULL(CAST(web_activity.current_transaction_web_traffic_device_browser AS STRING), 'UNKNOWN') AS current_transaction_web_traffic_device_browser,
IFNULL(CAST(web_activity.current_transaction_web_traffic_device_category AS STRING), 'UNKNOWN') AS current_transaction_web_traffic_device_category,
IFNULL(CAST(web_activity.current_transaction_web_traffic_campaign AS STRING), 'UNKNOWN') AS current_transaction_web_traffic_campaign,
IFNULL(CAST(web_activity.current_transaction_web_traffic_medium AS STRING), 'UNKNOWN') AS current_transaction_web_traffic_medium,
IFNULL(CAST(web_activity.current_transaction_web_traffic_source AS STRING), 'UNKNOWN') AS current_transaction_web_traffic_source,
IFNULL(CAST(web_activity.current_transaction_web_traffic_channel_grouping AS STRING), 'UNKNOWN') AS current_transaction_web_traffic_channel_grouping,
IFNULL(CAST(web_activity.current_transaction_web_traffic_timeonsite AS FLOAT64),0) AS current_transaction_web_traffic_timeonsite,
IFNULL(CAST(web_activity.current_transaction_web_traffic_total_visits AS FLOAT64),0) AS current_transaction_web_traffic_total_visits,
IFNULL(CAST(web_activity.current_transaction_web_traffic_new_visits AS FLOAT64),0) AS current_transaction_web_traffic_new_visits,
IFNULL(CAST(web_activity.current_transaction_web_traffic_pageviews AS FLOAT64),0) AS current_transaction_web_traffic_pageviews,
IFNULL(CAST(web_activity.current_transaction_web_traffic_hits AS FLOAT64),0) AS current_transaction_web_traffic_hits,
IFNULL(CAST(web_activity.current_transaction_web_traffic_bounces AS FLOAT64),0) AS current_transaction_web_traffic_bounces,
IFNULL(CAST(web_activity.current_transaction_web_traffic_product_page_event_count AS FLOAT64),0) AS current_transaction_web_traffic_product_page_event_count,
IFNULL(CAST(web_activity.current_transaction_web_traffic_productpage_timeonsite AS FLOAT64),0) AS current_transaction_web_traffic_productpage_timeonsite,
IFNULL(CAST(web_activity.current_transaction_web_traffic_checkoutprocess_timeonsite AS FLOAT64),0) AS current_transaction_web_traffic_checkoutprocess_timeonsite,
IFNULL(CAST(web_activity.current_transaction_web_traffic_clickthroughproductlists_events AS FLOAT64),0) AS current_transaction_web_traffic_clickthroughproductlists_events,
IFNULL(CAST(web_activity.current_transaction_web_traffic_productdetailviews_events AS FLOAT64),0) AS current_transaction_web_traffic_productdetailviews_events,
IFNULL(CAST(web_activity.current_transaction_web_traffic_producttocarts_events AS FLOAT64),0) AS current_transaction_web_traffic_producttocarts_events,
IFNULL(CAST(web_activity.current_transaction_web_traffic_removeproducts_events AS FLOAT64),0) AS current_transaction_web_traffic_removeproducts_events,
IFNULL(CAST(web_activity.current_transaction_web_traffic_checkout_events AS FLOAT64),0) AS current_transaction_web_traffic_checkout_events,
IFNULL(CAST(web_activity.current_transaction_web_traffic_confirmpurchases_events AS FLOAT64),0) AS current_transaction_web_traffic_confirmpurchases_events,
IFNULL(CAST(web_activity.current_transaction_web_traffic_checksize_events AS FLOAT64),0) AS current_transaction_web_traffic_checksize_events,

-- customer level CRM attributes & past customer purchase behaviors
IFNULL(CAST(crm_features.past_transaction_transaction_amt_sum AS FLOAT64),0) AS past_transaction_transaction_amt_sum,
IFNULL(CAST(crm_features.past_transaction_transaction_amt_mean AS FLOAT64),0) AS past_transaction_transaction_amt_mean,
crm_features.past_transaction_transaction_amt_median,
crm_features.past_transaction_transaction_amt_min,
crm_features.past_transaction_transaction_amt_max,
crm_features.past_transaction_transaction_amt_std,
IFNULL(CAST(crm_features.past_transaction_refund_product_quantity_sum AS FLOAT64),0) AS past_transaction_refund_product_quantity_sum,
IFNULL(CAST(crm_features.past_transaction_refund_product_quantity_mean AS FLOAT64),0) AS past_transaction_refund_product_quantity_mean,
crm_features.past_transaction_refund_product_quantity_median,
crm_features.past_transaction_refund_product_quantity_min,
crm_features.past_transaction_refund_product_quantity_max,
crm_features.past_transaction_refund_product_quantity_std,

IFNULL(CAST(crm_features.past_transaction_refund_amt_sum AS FLOAT64),0) AS past_transaction_refund_amt_sum,
IFNULL(CAST(crm_features.past_transaction_refund_amt_mean AS FLOAT64),0) AS past_transaction_refund_amt_mean,
crm_features.past_transaction_refund_amt_median,
crm_features.past_transaction_refund_amt_min,
crm_features.past_transaction_refund_amt_max,
crm_features.past_transaction_refund_amt_std,
IFNULL(CAST(crm_features.past_transaction_product_count_sum AS FLOAT64),0) AS past_transaction_product_count_sum,
IFNULL(CAST(crm_features.past_transaction_product_count_mean AS FLOAT64),0) AS past_transaction_product_count_mean,
crm_features.past_transaction_product_count_median,
crm_features.past_transaction_product_count_min,
crm_features.past_transaction_product_count_max,
crm_features.past_transaction_product_count_std,
IFNULL(CAST(crm_features.past_transaction_unique_product_count_sum AS FLOAT64),0) AS past_transaction_unique_product_count_sum,
IFNULL(CAST(crm_features.past_transaction_unique_product_count_mean AS FLOAT64),0) AS past_transaction_unique_product_count_mean,
crm_features.past_transaction_unique_product_count_median,
crm_features.past_transaction_unique_product_count_min,
crm_features.past_transaction_unique_product_count_max,
crm_features.past_transaction_unique_product_count_std,
IFNULL(CAST(crm_features.past_transaction_refund_product_count_sum AS FLOAT64),0) AS past_transaction_refund_product_count_sum,
IFNULL(CAST(crm_features.past_transaction_refund_product_count_mean AS FLOAT64),0) AS past_transaction_refund_product_count_mean,
crm_features.past_transaction_refund_product_count_median,
crm_features.past_transaction_refund_product_count_min,
crm_features.past_transaction_refund_product_count_max,
crm_features.past_transaction_refund_product_count_std,
IFNULL(CAST(crm_features.past_transaction_refund_unique_model_count_sum AS FLOAT64),0) AS past_transaction_refund_unique_model_count_sum,
IFNULL(CAST(crm_features.past_transaction_refund_unique_model_count_mean AS FLOAT64),0) AS past_transaction_refund_unique_model_count_mean,
crm_features.past_transaction_refund_unique_model_count_median,
crm_features.past_transaction_refund_unique_model_count_min,
crm_features.past_transaction_refund_unique_model_count_max,
crm_features.past_transaction_refund_unique_model_count_std,
IFNULL(CAST(crm_features.first_transaction_boolean AS FLOAT64),0) AS first_transaction_boolean,
IFNULL(CAST(crm_features.past_transaction_count AS FLOAT64),0) AS past_transaction_count,
IFNULL(CAST(crm_features.past_refund_count AS FLOAT64),0) AS past_refund_count,
IFNULL(CAST(crm_features.past_transaction_most_recent_refund_amt AS FLOAT64),0) AS past_transaction_most_recent_refund_amt,
IFNULL(CAST(crm_features.past_transaction_most_recent_transaction_amt AS FLOAT64),0) AS past_transaction_most_recent_transaction_amt,
CAST(crm_features.past_transaction_refund_recency_in_days_bucket AS STRING) AS past_transaction_refund_recency_in_days_bucket,
CAST(crm_features.past_transaction_refund_amount_percentage_bucket AS STRING) AS past_transaction_refund_amount_percentage_bucket,
CAST(crm_features.past_transaction_refund_count_percentage_bucket AS STRING) AS past_transaction_refund_count_percentage_bucket,
IFNULL(CAST(crm_features.customer_country AS STRING), 'UNKNOWN') AS customer_country,
IFNULL(CAST(crm_features.customer_most_frequent_payment_method AS STRING), 'UNKNOWN') AS customer_most_frequent_payment_method,
IFNULL(CAST(crm_features.customer_most_used_device AS STRING),'UNKNOWN') AS customer_most_used_device,
IFNULL(CAST(crm_features.customer_most_used_category AS STRING),'UNKNOWN') AS customer_most_used_category,
IFNULL(CAST(crm_features.registered_user_boolean AS FLOAT64),0) AS registered_user_boolean,
IFNULL(CAST(crm_features.customer_tenure_bucket AS STRING), 'UNKNOWN') AS customer_tenure_bucket,

--past same product refund history stats for the products in the current customer's basket
past_same_product_refund_stats.product_refund_rate_min,
past_same_product_refund_stats.product_refund_rate_max,
past_same_product_refund_stats.product_refund_rate_mean,
past_same_product_refund_stats.estimated_product_refund_amt_sum,

--past same customer & same product refund history stats
past_same_customer_same_product_refund_stats.same_product_same_customer_past_unique_transaction_count,
past_same_customer_same_product_refund_stats.same_product_same_customer_past_unique_refund_transaction_count,
past_same_customer_same_product_refund_stats.estimated_refund_value_based_on_customer_past_refund_history
FROM `{project_id}.{dataset_id}.target_variable_data`  target_variable
LEFT JOIN `{project_id}.{dataset_id}.current_transaction_feature` current_transaction_attributes
ON target_variable.transaction_id = current_transaction_attributes.transaction_id
LEFT JOIN `{project_id}.{dataset_id}.web_activity_preprocessed_table` web_activity
ON web_activity.transaction_id = target_variable.transaction_id
LEFT JOIN `{project_id}.{dataset_id}.customer_level_features_data`  crm_features
ON crm_features.transaction_id = target_variable.transaction_id
LEFT JOIN `{project_id}.{dataset_id}.transaction_level_past_same_product_refund_stats`  past_same_product_refund_stats
ON past_same_product_refund_stats.transaction_id = target_variable.transaction_id
LEFT JOIN `{project_id}.{dataset_id}.transaction_level_same_customer_same_product_past_transactions_refund_staging_data` past_same_customer_same_product_refund_stats 
ON past_same_customer_same_product_refund_stats.transaction_id = target_variable.transaction_id
WHERE crm_features.customer_email_hash IS NOT NULL 
AND target_variable.transaction_id IS NOT NULL;







