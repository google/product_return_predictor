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

-- Build predict datasets for a machine learning model for predict transation refund amount.
-- This script creates two datasets final modeling data for first-time transaction and final modeling data for non first-time transaction

CREATE OR REPLACE TABLE
  `{project_id}.{dataset_id}.final_modeling_for_non_first_time_transaction_df` AS
  
SELECT 
CASE WHEN ABS(MOD(FARM_FINGERPRINT(TO_JSON_STRING(STRUCT(transaction_id))), 100)) BETWEEN 0 AND 15 THEN 'TEST' 
WHEN ABS(MOD(FARM_FINGERPRINT(TO_JSON_STRING(STRUCT(transaction_id))), 100)) BETWEEN 15 AND 30 THEN 'VALIDATE'  
WHEN ABS(MOD(FARM_FINGERPRINT(TO_JSON_STRING(STRUCT(transaction_id))), 100)) BETWEEN 30 AND 100 THEN 'TRAIN' END AS predefined_split_column,
transaction_id  ,
-- AutoML having trouble with datetime datatype
--transaction_date  ,
CAST(EXTRACT(Month FROM transaction_date) AS STRING) AS transaction_month,
CAST(FORMAT_DATE('%a', transaction_date) AS STRING) AS transaction_day_of_the_week,
customer_email_hash ,
transaction_amt ,
refund_amt  ,
discount_type ,
transaction_delivery_type ,
transaction_payment_type  ,
transaction_total_product_quantity  ,
transaction_shipping_cost ,
freeshipping_boolean  ,
nunique_product_categories  ,
nunique_product_names ,
transaction_level_max_product_quantity  ,
transaction_level_duplicate_product_boolean ,
transaction_level_same_product_max_unique_product_size_count  ,
transaction_level_same_product_different_size_boolean ,
avg_product_price ,
current_transaction_web_traffic_device_browser  ,
current_transaction_web_traffic_device_category ,
current_transaction_web_traffic_campaign  ,
current_transaction_web_traffic_medium  ,
current_transaction_web_traffic_source  ,
current_transaction_web_traffic_channel_grouping  ,
current_transaction_web_traffic_timeonsite  ,
current_transaction_web_traffic_total_visits  ,
current_transaction_web_traffic_new_visits  ,
current_transaction_web_traffic_pageviews ,
current_transaction_web_traffic_hits  ,
current_transaction_web_traffic_bounces ,
current_transaction_web_traffic_product_page_event_count  ,
current_transaction_web_traffic_productpage_timeonsite  ,
current_transaction_web_traffic_checkoutprocess_timeonsite  ,
current_transaction_web_traffic_clickthroughproductlists_events ,
current_transaction_web_traffic_productdetailviews_events ,
current_transaction_web_traffic_producttocarts_events ,
current_transaction_web_traffic_removeproducts_events ,
current_transaction_web_traffic_checkout_events ,
current_transaction_web_traffic_confirmpurchases_events ,
current_transaction_web_traffic_checksize_events  ,
past_transaction_transaction_amt_sum  ,
past_transaction_transaction_amt_mean ,
past_transaction_transaction_amt_median ,
past_transaction_transaction_amt_min  ,
past_transaction_transaction_amt_max  ,
IFNULL(past_transaction_transaction_amt_std, 0)  AS past_transaction_transaction_amt_std,
past_transaction_refund_product_quantity_sum  ,
past_transaction_refund_product_quantity_mean ,
past_transaction_refund_product_quantity_median ,
past_transaction_refund_product_quantity_min  ,
past_transaction_refund_product_quantity_max  ,
IFNULL(past_transaction_refund_product_quantity_std,0) AS  past_transaction_refund_product_quantity_std,
past_transaction_refund_amt_sum ,
past_transaction_refund_amt_mean  ,
past_transaction_refund_amt_median  ,
past_transaction_refund_amt_min ,
past_transaction_refund_amt_max ,
IFNULL(past_transaction_refund_amt_std,0) AS past_transaction_refund_amt_std,
past_transaction_product_count_sum  ,
past_transaction_product_count_mean ,
past_transaction_product_count_median ,
past_transaction_product_count_min  ,
past_transaction_product_count_max  ,
IFNULL(past_transaction_product_count_std,0) AS past_transaction_product_count_std,
past_transaction_unique_product_count_sum ,
past_transaction_unique_product_count_mean  ,
past_transaction_unique_product_count_median  ,
past_transaction_unique_product_count_min ,
past_transaction_unique_product_count_max ,
IFNULL(past_transaction_unique_product_count_std,0) AS past_transaction_unique_product_count_std ,
past_transaction_refund_product_count_sum ,
past_transaction_refund_product_count_mean  ,
past_transaction_refund_product_count_median  ,
past_transaction_refund_product_count_min ,
past_transaction_refund_product_count_max ,
IFNULL(past_transaction_refund_product_count_std,0) AS past_transaction_refund_product_count_std ,
past_transaction_refund_unique_model_count_sum  ,
past_transaction_refund_unique_model_count_mean ,
past_transaction_refund_unique_model_count_median ,
past_transaction_refund_unique_model_count_min  ,
past_transaction_refund_unique_model_count_max  ,
IFNULL(past_transaction_refund_unique_model_count_std,0) AS past_transaction_refund_unique_model_count_std,
past_transaction_count  ,
past_refund_count ,
past_transaction_most_recent_refund_amt ,
past_transaction_most_recent_transaction_amt  ,
past_transaction_refund_recency_in_days_bucket  ,
past_transaction_refund_amount_percentage_bucket  ,
past_transaction_refund_count_percentage_bucket ,
customer_country  ,
customer_most_frequent_payment_method ,
customer_most_used_device ,
customer_most_used_category ,
registered_user_boolean ,
customer_tenure_bucket  ,
product_refund_rate_min ,
product_refund_rate_max ,
product_refund_rate_mean  ,
estimated_product_refund_amt_sum  ,
same_product_same_customer_past_unique_transaction_count  ,
same_product_same_customer_past_unique_refund_transaction_count ,
estimated_refund_value_based_on_customer_past_refund_history  
FROM `{project_id}.{dataset_id}.final_modeling_df`
WHERE past_transaction_refund_amount_percentage_bucket <> 'no past transaction';

CREATE OR REPLACE TABLE
  `{project_id}.{dataset_id}.final_modeling_for_firsttime_transaction_df` AS
  
SELECT 
CASE WHEN ABS(MOD(FARM_FINGERPRINT(TO_JSON_STRING(STRUCT(transaction_id))), 100)) BETWEEN 0 AND 15 THEN 'TEST' 
WHEN ABS(MOD(FARM_FINGERPRINT(TO_JSON_STRING(STRUCT(transaction_id))), 100)) BETWEEN 15 AND 30 THEN 'VALIDATE'  
WHEN ABS(MOD(FARM_FINGERPRINT(TO_JSON_STRING(STRUCT(transaction_id))), 100)) BETWEEN 30 AND 100 THEN 'TRAIN' END AS predefined_split_column,
transaction_id  ,
CAST(EXTRACT(Month FROM transaction_date) AS STRING) AS transaction_month,
CAST(FORMAT_DATE('%a', transaction_date) AS STRING) AS transaction_day_of_the_week  ,
customer_email_hash ,
transaction_amt ,
refund_amt  ,
CASE WHEN discount_type='' THEN 'No Discount' ELSE discount_type END AS discount_type,
transaction_delivery_type ,
transaction_payment_type  ,
transaction_total_product_quantity  ,
transaction_shipping_cost ,
freeshipping_boolean  ,
nunique_product_categories  ,
nunique_product_names ,
transaction_level_max_product_quantity  ,
transaction_level_duplicate_product_boolean ,
transaction_level_same_product_max_unique_product_size_count  ,
transaction_level_same_product_different_size_boolean ,
avg_product_price ,
current_transaction_web_traffic_device_browser  ,
current_transaction_web_traffic_device_category ,
current_transaction_web_traffic_campaign  ,
current_transaction_web_traffic_medium  ,
current_transaction_web_traffic_source  ,
current_transaction_web_traffic_channel_grouping  ,
current_transaction_web_traffic_timeonsite  ,
current_transaction_web_traffic_total_visits  ,
current_transaction_web_traffic_new_visits  ,
current_transaction_web_traffic_pageviews ,
current_transaction_web_traffic_hits  ,
current_transaction_web_traffic_bounces ,
current_transaction_web_traffic_product_page_event_count  ,
current_transaction_web_traffic_productpage_timeonsite  ,
current_transaction_web_traffic_checkoutprocess_timeonsite  ,
current_transaction_web_traffic_clickthroughproductlists_events ,
current_transaction_web_traffic_productdetailviews_events ,
current_transaction_web_traffic_producttocarts_events ,
current_transaction_web_traffic_removeproducts_events ,
current_transaction_web_traffic_checkout_events ,
current_transaction_web_traffic_confirmpurchases_events ,
current_transaction_web_traffic_checksize_events  ,
customer_country  ,
customer_most_frequent_payment_method ,
customer_most_used_device ,
customer_most_used_category ,
registered_user_boolean ,
customer_tenure_bucket  ,
product_refund_rate_min ,
product_refund_rate_max ,
product_refund_rate_mean  ,
estimated_product_refund_amt_sum  
FROM `{project_id}.{dataset_id}.final_modeling_df`
WHERE past_transaction_refund_amount_percentage_bucket = 'no past transaction';





