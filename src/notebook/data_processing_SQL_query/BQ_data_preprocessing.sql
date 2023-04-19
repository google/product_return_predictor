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

-- Build transaction id to the customer email hash mapping data

CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.transaction_to_email_hash_mapping` AS
    SELECT DISTINCT EMAIL_HASH, CAST(ID_PEDIDO_ECOMMERCE AS STRING) as transaction_id
    FROM `{source_table_project_id}.{source_table_dataset_id}.{source_table_transaction_id_to_customer_id_mapping_table_name}`;

-- Build preprocessing transaction data

CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.preprocessed_transaction_data` AS
  SELECT
    PARSE_DATE('%Y%m%d', date) AS transaction_date,
    CAST(hits.item.transactionid AS STRING) AS transaction_id,
    CAST(hits.transaction.transactionRevenue AS FLOAT64)/1000000 AS transaction_amt,
    CAST(prod.productQuantity AS FLOAT64) AS total_product_quantity,
    CAST(prod.productPrice AS FLOAT64)/1000000 AS product_price,
    NULLIF(prod.productSKU, '') AS productsku,
    CAST((
      SELECT
        value
      FROM
        UNNEST(prod.customdimensions)
      WHERE
        INDEX = 97
      GROUP BY
        value) AS INT64) AS product_size,
    CAST(prod.productRevenue AS FLOAT64)/1000000 AS product_revenue,
    NULLIF(prod.v2ProductCategory, '') AS product_category,
    NULLIF(prod.v2ProductName, '') AS product_name,
    NULLIF((
      SELECT
        value
      FROM
        UNNEST(hits.customdimensions)
      WHERE
        INDEX = 41
      GROUP BY
        value), 'UNKNOWN') AS discount_type,
    NULLIF((
      SELECT
        value
      FROM
        UNNEST(hits.customdimensions)
      WHERE
        INDEX = 33
      GROUP BY
        value), 'UNKNOWN') AS transaction_delivery_type,
    IFNULL((CAST(hits.transaction.transactionShipping AS FLOAT64)/1000000),0.0) AS transaction_shipping_cost,
    (
    SELECT
      value
    FROM
      UNNEST(hits.customdimensions)
    WHERE
      INDEX = 36
    GROUP BY
      value) AS transaction_payment_type
  FROM
    `{source_table_project_id}.{source_table_dataset_id}.{source_table_web_analytics_table_name}`,
    UNNEST(hits) AS hits,
    UNNEST(hits.product) AS prod
  WHERE
    hits.item.transactionid IS NOT NULL
    AND CAST(hits.item.transactionid AS FLOAT64) > 0
    AND CAST(hits.transaction.transactionRevenue AS FLOAT64)>0;


-- Create transaction refund preprocessed data

CREATE OR REPLACE TABLE
  `{project_id}.{dataset_id}.refund_preprocessed_table` AS
SELECT
  *,
  CASE
    WHEN (refund_product_quantity>0) AND (refund_amt>0) THEN 1
  ELSE
  0
END
  AS refund_boolean,
  refund_amt/transaction_revenue AS refund_rate
FROM (
  SELECT
    CAST(TRANSACTION_ID AS STRING) AS transaction_id,
    MAX(IFNULL(CAST(TRANSACTION_REVENUE AS FLOAT64),0)) AS transaction_revenue,
    SUM(CAST(PRODUCT_REFUND_QUANTITY AS FLOAT64))*-1.0 AS refund_product_quantity,
    SUM(CAST(PRODUCT_REEFUND_VALUE AS FLOAT64))*-1.0 AS refund_amt,
    COUNT(PRODUCT_NAME) AS refund_product_count,
    COUNT(DISTINCT PRODUCT_MODEL) AS refund_unique_model_count
  FROM
    `{source_table_project_id}.{source_table_dataset_id}.{source_table_refund_data_table_name}`
  WHERE
    APLICATION_TYPE = 'Web Responsive'
    AND CAST(PRODUCT_REFUND_QUANTITY AS FLOAT64)<0
  GROUP BY
    TRANSACTION_ID) transaction_level_refund_data
  WHERE transaction_revenue>0 AND refund_amt>0;


-- Create refund transaction preprocessed data
CREATE OR REPLACE TABLE
  `{project_id}.{dataset_id}.refund_transaction_preprocessed_table` AS
(
SELECT
transaction_table.transaction_date,
transaction_table.transaction_id,
IFNULL(transaction_table.transaction_amt, 0) AS transaction_amt,
IFNULL(transaction_table.transaction_unique_product_count, 0) AS transaction_unique_product_count,
IFNULL(transaction_table.transaction_product_count, 0) AS transaction_product_count,
IFNULL(refund_table.refund_product_quantity,0) AS refund_product_quantity,
IFNULL(refund_table.refund_amt, 0) AS refund_amt,
IFNULL(refund_table.refund_product_count, 0) AS refund_product_count,
IFNULL(refund_table.refund_unique_model_count, 0) AS refund_unique_model_count
FROM (
SELECT transaction_id,
MIN(transaction_date) AS transaction_date,
MAX(transaction_amt) AS transaction_amt,
COUNT(product_name) AS transaction_product_count,
COUNT(DISTINCT product_name) AS transaction_unique_product_count,
FROM `{project_id}.{dataset_id}.preprocessed_transaction_data`
WHERE transaction_id IS NOT NULL
AND CAST(transaction_id AS FLOAT64)>0
GROUP BY 1
) transaction_table
LEFT JOIN `{project_id}.{dataset_id}.refund_preprocessed_table` refund_table
ON refund_table.transaction_id = transaction_table.transaction_id
WHERE transaction_table.transaction_amt>0
);

-- Create CRM preprocessing data (transaction id to customer email hash mapping data)
CREATE OR REPLACE TABLE
  `{project_id}.{dataset_id}.crm_preprocessed_table` AS
SELECT
  customer_email_hash,
  transaction_id,
  customer_country,
  customer_most_frequent_payment_method,
  customer_most_used_device,
  customer_most_used_category,
  REGISTRATION
FROM (
  SELECT
    EMAIL AS customer_email_hash,
    COUNTRY_ISO AS customer_country,
    PAYMENT_METHOD AS customer_most_frequent_payment_method,
    MOST_USED_DEVICE AS customer_most_used_device,
    MOST_USED_CATEGORY AS customer_most_used_category,
    REGISTRATION,
    email_hash_to_transaction_mapping.transaction_id,
    ROW_NUMBER() OVER (PARTITION BY EMAIL ORDER BY EMAIL) AS ROW
  FROM
    `{source_table_project_id}.{source_table_dataset_id}.{source_crm_table_name}` crm_data
  INNER JOIN
    `{project_id}.{dataset_id}.transaction_to_email_hash_mapping` email_hash_to_transaction_mapping
  ON
    email_hash_to_transaction_mapping.EMAIL_HASH = crm_data.EMAIL ) X
WHERE
  ROW=1;

-- Create customer level transaction refund preprocessed data

CREATE OR REPLACE TABLE
  `{project_id}.{dataset_id}.customer_level_transaction_refund_preprocessed_table` AS
(
SELECT
EMAIL_HASH,
transaction_refund.transaction_id,
transaction_refund.transaction_date,
transaction_refund.transaction_amt,
transaction_refund.transaction_product_count,
transaction_refund.transaction_unique_product_count,
transaction_refund.refund_product_quantity,
transaction_refund.refund_amt,
transaction_refund.refund_product_count,
transaction_refund.refund_unique_model_count
FROM `{project_id}.{dataset_id}.transaction_to_email_hash_mapping` transaction_to_email_hash_mapping
INNER JOIN `{project_id}.{dataset_id}.refund_transaction_preprocessed_table` transaction_refund
ON transaction_refund.transaction_id = transaction_to_email_hash_mapping.transaction_id
);




-- Create web analytics table

CREATE OR REPLACE TABLE
  `{project_id}.{dataset_id}.web_activity_preprocessed_table` AS
WITH
  all_session_ids_with_transactions AS (
  SELECT
    DISTINCT CONCAT(fullvisitorid,'_',CAST(visitstarttime AS string)) AS session_id,
  FROM
    `{source_table_project_id}.{source_table_dataset_id}.{source_table_web_analytics_table_name}`,
    UNNEST(hits) AS hits
  WHERE
    CAST(hits.item.transactionid AS FLOAT64) IS NOT NULL
    AND CAST(hits.item.transactionid AS FLOAT64) > 0
    AND CAST(hits.transaction.transactionRevenue AS FLOAT64)>0 ),
  session_level_website_activity AS (
  SELECT
    CONCAT(fullvisitorid,'_',CAST(visitstarttime AS string)) AS session_id,
    ANY_VALUE(NULLIF(device.browser, '(none)')) AS current_transaction_web_traffic_device_browser,
    ANY_VALUE(NULLIF(device.deviceCategory, '(none)')) AS current_transaction_web_traffic_device_category,
    ANY_VALUE(NULLIF(trafficSource.campaign, '(none)')) AS current_transaction_web_traffic_campaign,
    ANY_VALUE(NULLIF(trafficSource.medium, '(none)')) AS current_transaction_web_traffic_medium,
    ANY_VALUE(NULLIF(trafficSource.source, '(none)')) AS current_transaction_web_traffic_source,
    ANY_VALUE(NULLIF(channelGrouping, '(none)')) AS current_transaction_web_traffic_channel_grouping,
    SUM(IFNULL(totals.timeOnSite,0)) AS current_transaction_web_traffic_timeonsite,
    SUM(IFNULL(totals.visits, 0)) AS current_transaction_web_traffic_total_visits,
    SUM(IFNULL(totals.newVisits, 0)) AS current_transaction_web_traffic_new_visits,
    SUM(IFNULL(totals.pageviews, 0)) AS current_transaction_web_traffic_pageviews,
    SUM(IFNULL(totals.hits,0)) AS current_transaction_web_traffic_hits,
    SUM(IFNULL(totals.bounces,0)) AS current_transaction_web_traffic_bounces,
    SUM(CASE
        WHEN hits.eventInfo.eventCategory = 'ficha_producto' THEN 1
      ELSE
      0
    END
      ) AS current_transaction_web_traffic_product_page_event_count,
    SUM(CASE
        WHEN CAST(hits.eCommerceAction.action_type AS INT) >=1 AND CAST(hits.eCommerceAction.action_type AS INT) <=6 THEN IFNULL(totals.timeOnSite,0)
      ELSE
      0
    END
      ) AS current_transaction_web_traffic_productpage_timeonsite,
    SUM(CASE
        WHEN hits.eventInfo.eventCategory = 'ficha_producto' THEN IFNULL(totals.timeOnSite,0)
      ELSE
      0
    END
      ) AS current_transaction_web_traffic_checkoutprocess_timeonsite,
    SUM(CASE
        WHEN hits.eCommerceAction.action_type = '1' THEN 1
      ELSE
      0
    END
      ) AS current_transaction_web_traffic_clickthroughproductlists_events,
    SUM(CASE
        WHEN hits.eCommerceAction.action_type = '2' THEN 1
      ELSE
      0
    END
      ) AS current_transaction_web_traffic_productdetailviews_events,
    SUM(CASE
        WHEN hits.eCommerceAction.action_type = '3' THEN 1
      ELSE
      0
    END
      ) AS current_transaction_web_traffic_producttocarts_events,
    SUM(CASE
        WHEN hits.eCommerceAction.action_type = '4' THEN 1
      ELSE
      0
    END
      ) AS current_transaction_web_traffic_removeproducts_events,
    SUM(CASE
        WHEN hits.eCommerceAction.action_type = '5' THEN 1
      ELSE
      0
    END
      ) AS current_transaction_web_traffic_checkout_events,
    SUM(CASE
        WHEN hits.eCommerceAction.action_type = '6' THEN 1
      ELSE
      0
    END
      ) AS current_transaction_web_traffic_confirmpurchases_events,
    SUM(CASE
        WHEN hits.eventInfo.eventCategory = 'ficha_producto' AND REGEXP_CONTAINS(hits.eventInfo.eventAction, r'ver_medidas|ver_guia_tallas_fit') THEN 1
      ELSE
      0
    END
      ) AS current_transaction_web_traffic_checksize_events
  FROM
    `{source_table_project_id}.{source_table_dataset_id}.{source_table_web_analytics_table_name}`,
    UNNEST(hits) AS hits
  INNER JOIN
    all_session_ids_with_transactions session_id_table
  ON
    session_id_table.session_id = CONCAT(fullvisitorid,'_',CAST(visitstarttime AS string))
  GROUP BY
    1 ),
  transaction_id_to_session_id_mapping AS (
  SELECT
    DISTINCT CONCAT(fullvisitorid,'_',CAST(visitstarttime AS string)) AS session_id,
    hits.item.transactionid AS transaction_id
  FROM
    `{source_table_project_id}.{source_table_dataset_id}.{source_table_web_analytics_table_name}`,
    UNNEST(hits) AS hits
  WHERE
    CAST(hits.item.transactionid AS FLOAT64) IS NOT NULL
    AND CAST(hits.item.transactionid AS FLOAT64) > 0
    AND CAST(hits.transaction.transactionRevenue AS FLOAT64)>0 )
SELECT
  CAST(transaction_id_back_bone_table.transaction_id AS STRING) AS transaction_id,
  IFNULL(ANY_VALUE(current_transaction_web_traffic_device_browser), 'UNKNOWN') AS current_transaction_web_traffic_device_browser,
  IFNULL(ANY_VALUE(current_transaction_web_traffic_device_category), 'UNKNOWN') AS current_transaction_web_traffic_device_category,
  IFNULL(ANY_VALUE(current_transaction_web_traffic_campaign), 'UNKNOWN') AS current_transaction_web_traffic_campaign,
  IFNULL(ANY_VALUE(current_transaction_web_traffic_medium), 'UNKNOWN') AS current_transaction_web_traffic_medium,
  IFNULL(ANY_VALUE(current_transaction_web_traffic_source), 'UNKNOWN') AS current_transaction_web_traffic_source,
  IFNULL(ANY_VALUE(current_transaction_web_traffic_channel_grouping), 'UNKNOWN') AS current_transaction_web_traffic_channel_grouping,
  SUM(IFNULL(current_transaction_web_traffic_timeonsite,0)) AS current_transaction_web_traffic_timeonsite,
  SUM(IFNULL(current_transaction_web_traffic_total_visits,0)) AS current_transaction_web_traffic_total_visits,
  SUM(IFNULL(current_transaction_web_traffic_new_visits,0)) AS current_transaction_web_traffic_new_visits,
  SUM(IFNULL(current_transaction_web_traffic_pageviews,0)) AS current_transaction_web_traffic_pageviews,
  SUM(IFNULL(current_transaction_web_traffic_hits,0)) AS current_transaction_web_traffic_hits,
  SUM(IFNULL(current_transaction_web_traffic_bounces,0)) AS current_transaction_web_traffic_bounces,
  SUM(IFNULL(current_transaction_web_traffic_product_page_event_count,0)) AS current_transaction_web_traffic_product_page_event_count,
  SUM(IFNULL(current_transaction_web_traffic_productpage_timeonsite,0)) AS current_transaction_web_traffic_productpage_timeonsite,
  SUM(IFNULL(current_transaction_web_traffic_checkoutprocess_timeonsite,0)) AS current_transaction_web_traffic_checkoutprocess_timeonsite,
  SUM(IFNULL(current_transaction_web_traffic_clickthroughproductlists_events,0)) AS current_transaction_web_traffic_clickthroughproductlists_events,
  SUM(IFNULL(current_transaction_web_traffic_productdetailviews_events,0)) AS current_transaction_web_traffic_productdetailviews_events,
  SUM(IFNULL(current_transaction_web_traffic_producttocarts_events,0)) AS current_transaction_web_traffic_producttocarts_events,
  SUM(IFNULL(current_transaction_web_traffic_removeproducts_events,0)) AS current_transaction_web_traffic_removeproducts_events,
  SUM(IFNULL(current_transaction_web_traffic_checkout_events,0)) AS current_transaction_web_traffic_checkout_events,
  SUM(IFNULL(current_transaction_web_traffic_confirmpurchases_events,0)) AS current_transaction_web_traffic_confirmpurchases_events,
  SUM(IFNULL(current_transaction_web_traffic_checksize_events,0)) AS current_transaction_web_traffic_checksize_events
FROM
  transaction_id_to_session_id_mapping transaction_id_back_bone_table
LEFT JOIN
  session_level_website_activity session_table
ON
  transaction_id_back_bone_table.session_id = session_table.session_id
GROUP BY
  1 ;

-- Create customer past purchase history feature data

CREATE OR REPLACE TABLE
  `{project_id}.{dataset_id}.customer_past_purchase_history_features_staging_data` AS

SELECT
current_transaction.EMAIL_HASH AS customer_email_hash,
current_transaction.transaction_id AS transaction_id,
current_transaction.transaction_date AS current_transaction_date,
past_transaction.transaction_date AS past_transaction_date,
IFNULL(past_transaction.transaction_amt, 0) AS past_transaction_transaction_amt,
IFNULL(past_transaction.refund_product_quantity, 0) AS past_transaction_refund_product_quantity,
IFNULL(past_transaction.transaction_unique_product_count, 0) AS past_transaction_unique_product_count,
IFNULL(past_transaction.transaction_product_count, 0) AS past_transaction_product_count,
IFNULL(past_transaction.refund_product_count, 0) AS past_transaction_refund_product_count,
IFNULL(past_transaction.refund_unique_model_count, 0) AS past_transaction_refund_unique_model_count,
IFNULL(past_transaction.refund_amt, 0) AS past_transaction_refund_amt
FROM `{project_id}.{dataset_id}.customer_level_transaction_refund_preprocessed_table` current_transaction
LEFT JOIN `{project_id}.{dataset_id}.customer_level_transaction_refund_preprocessed_table` past_transaction
ON current_transaction.EMAIL_HASH = past_transaction.EMAIL_HASH
AND current_transaction.transaction_date > past_transaction.transaction_date;

-- Create product level transaction refund preprocessed data

CREATE OR REPLACE TABLE
  `{project_id}.{dataset_id}.product_level_transactions_refund_staging_data` AS

WITH product_level_transaction AS (
SELECT
IFNULL(productsku, 'UNKNOWN') AS productsku,
transaction_date,
COUNT(DISTINCT transaction_id) AS unique_transaction_count
FROM `{project_id}.{dataset_id}.preprocessed_transaction_data`
GROUP BY 1,2
HAVING MAX(transaction_amt)>0
),
product_level_refund AS (
SELECT
NULLIF(PRODUCT_SKU, '') AS productsku,
DATE(DATE) as transaction_date,
COUNT(DISTINCT TRANSACTION_ID) AS unique_refund_transaction_count
FROM `{source_table_project_id}.{source_table_dataset_id}.{source_table_refund_data_table_name}`
WHERE APLICATION_TYPE = 'Web Responsive'
AND CAST(PRODUCT_REFUND_QUANTITY AS FLOAT64)<0
GROUP BY 1,2
)
SELECT product_level_transaction.productsku,
product_level_transaction.transaction_date,
product_level_transaction.unique_transaction_count as past_unique_transaction_count,
IFNULL(unique_refund_transaction_count,0) AS past_unique_refund_transaction_count
FROM product_level_transaction
LEFT JOIN product_level_refund
ON product_level_transaction.productsku = product_level_refund.productsku
AND product_level_transaction.transaction_date = product_level_refund.transaction_date;


-- Create table for transaction product level quantity

CREATE OR REPLACE TABLE
  `{project_id}.{dataset_id}.transaction_product_level_quantity_staging_data` AS
WITH
  min_transaction_date AS (
  SELECT
    transaction_id,
    MIN(transaction_date) AS transaction_date
  FROM
    `{project_id}.{dataset_id}.preprocessed_transaction_data`
  GROUP BY
    1 ),
  transaction_product_level_data AS (
  SELECT
    transaction_id,
    IFNULL(product_name, 'UNKNOWN') AS product_name,
    IFNULL(product_category, 'UNKNOWN') AS product_category,
    product_size,
    productsku,
    AVG(product_price) AS product_price,
    SUM(total_product_quantity) AS transaction_total_product_quantity,
    SUM(product_revenue) AS product_revenue
  FROM
    `{project_id}.{dataset_id}.preprocessed_transaction_data`
  GROUP BY
    1,2,3,4,5)
SELECT
  transaction_product_level_data.*,
  min_transaction_date.transaction_date
FROM
  transaction_product_level_data
LEFT JOIN
  min_transaction_date
ON
  transaction_product_level_data.transaction_id = min_transaction_date.transaction_id ;

-- Create product level past transaction refund stats data

CREATE OR REPLACE TABLE
  `{project_id}.{dataset_id}.transaction_level_same_product_past_transaction_refund_stats_staging_data` AS
WITH
  product_level_revenue AS (
  SELECT
    transaction_id,
    productsku,
    SUM(IFNULL(product_revenue,0)) AS product_revenue
  FROM
    `{project_id}.{dataset_id}.transaction_product_level_quantity_staging_data`
  GROUP BY
    1,
    2),
  transaction_level_same_product_sku_past_transaction_count AS (
  SELECT
    current_transaction.transaction_id,
    current_transaction.productsku,
    MIN(current_transaction.transaction_date) AS transaction_date,
    MIN(current_transaction.transaction_total_product_quantity) AS current_transaction_product_quantity,
    SUM(IFNULL(historical_transaction.past_unique_transaction_count,0)) AS same_product_past_unique_transaction_count,
    SUM(IFNULL(historical_transaction.past_unique_refund_transaction_count,0)) AS same_product_past_unique_refund_transaction_count
  FROM
    `{project_id}.{dataset_id}.transaction_product_level_quantity_staging_data` current_transaction
  LEFT JOIN
    `{project_id}.{dataset_id}.product_level_transactions_refund_staging_data` historical_transaction
  ON
    current_transaction.productsku = historical_transaction.productsku
    AND current_transaction.transaction_date>historical_transaction.transaction_date
  GROUP BY
    1,
    2 ),
  product_level_past_refund_rate_data AS (
  SELECT
    transaction_id,
    productsku,
    transaction_date,
    current_transaction_product_quantity,
    same_product_past_unique_transaction_count,
    CASE
      WHEN same_product_past_unique_refund_transaction_count>same_product_past_unique_transaction_count THEN same_product_past_unique_transaction_count
    ELSE
    same_product_past_unique_refund_transaction_count
  END
    AS same_product_past_unique_refund_transaction_count
  FROM
    transaction_level_same_product_sku_past_transaction_count ),
  grand_refund_avg_data AS (
  SELECT
    transaction_date,
    SUM(past_unique_refund_transaction_count)/SUM(past_unique_transaction_count) AS grand_avg_refund_rate
  FROM
    `{project_id}.{dataset_id}.product_level_transactions_refund_staging_data`
  GROUP BY
    transaction_date ),
  calculated_product_refund_data AS (
  SELECT
    product_level_past_refund_rate_data.*,
    IFNULL(product_level_revenue.product_revenue,0) AS product_revenue,
    (same_product_past_unique_refund_transaction_count+grand_avg_refund_rate*0.2)/(same_product_past_unique_transaction_count+0.2) AS product_refund_rate
  FROM
    product_level_past_refund_rate_data
  LEFT JOIN
    product_level_revenue
  ON
    product_level_past_refund_rate_data.transaction_id = product_level_revenue.transaction_id
    AND product_level_past_refund_rate_data.productsku = product_level_revenue.productsku
  LEFT JOIN
    grand_refund_avg_data
  ON
    grand_refund_avg_data.transaction_date = product_level_past_refund_rate_data.transaction_date )
SELECT
  *,
  product_refund_rate * IFNULL(product_revenue,0) AS estimated_product_refund_amt
FROM
  calculated_product_refund_data;


-- Create transaction level attributes staging data

CREATE OR REPLACE TABLE
  `{project_id}.{dataset_id}.transaction_level_attributes_staging_data` AS

SELECT
transaction_id,
IFNULL(ANY_VALUE(discount_type), 'UNKNOWN') AS discount_type,
IFNULL(ANY_VALUE(transaction_delivery_type), 'UNKNOWN') AS transaction_delivery_type,
IFNULL(ANY_VALUE(transaction_payment_type), 'UNKNOWN') AS transaction_payment_type,
MIN(transaction_date) AS transaction_date,
MAX(transaction_amt) AS transaction_amt,
SUM(total_product_quantity) AS transaction_total_product_quantity,
MAX(transaction_shipping_cost) AS transaction_shipping_cost,
IF(MAX(transaction_shipping_cost)>0, 0, 1) AS freeshipping_boolean,
COUNT(DISTINCT product_category) AS nunique_product_categories,
COUNT(DISTINCT product_name) AS nunique_product_names
FROM `{project_id}.{dataset_id}.preprocessed_transaction_data`
GROUP BY 1;

-- Create table for target variable

CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.target_variable_data`
AS
SELECT
transaction_date,
transaction_table.transaction_id,
IFNULL(transaction_table.transaction_amt, 0) AS transaction_amt,
IFNULL(refund_table.refund_product_quantity,0) AS refund_product_quantity,
IFNULL(refund_table.refund_amt, 0) AS refund_amt,
IFNULL(refund_table.refund_boolean, 0) AS refund_boolean,
IFNULL(refund_rate,0) AS refund_rate
FROM
(
SELECT transaction_id,
MIN(transaction_date) AS transaction_date,
MAX(transaction_amt) AS transaction_amt
FROM `{project_id}.{dataset_id}.preprocessed_transaction_data`
WHERE transaction_id IS NOT NULL
AND CAST(transaction_id AS FLOAT64)>0
GROUP BY transaction_id
) transaction_table
LEFT JOIN  `{project_id}.{dataset_id}.refund_preprocessed_table`  refund_table
ON refund_table.transaction_id = transaction_table.transaction_id;



-- Create transaction level same customer same product past transactions refund staging table

CREATE OR REPLACE TABLE
  `{project_id}.{dataset_id}.transaction_level_same_customer_same_product_past_transactions_refund_staging_data` AS
WITH
  transaction_customer_product_level_data AS (
  SELECT
    customer_level_transaction_refund_data.transaction_id,
    customer_level_transaction_refund_data.EMAIL_HASH,
    transaction_product_level_quantity_data.productsku,
    transaction_product_level_quantity_data.transaction_date,
    transaction_product_level_quantity_data.product_price,
    transaction_product_level_quantity_data.transaction_total_product_quantity,
    transaction_product_level_quantity_data.product_revenue
  FROM
    `{project_id}.{dataset_id}.customer_level_transaction_refund_preprocessed_table` customer_level_transaction_refund_data
  LEFT JOIN (
    SELECT
      transaction_id,
      productsku,
      MIN(transaction_date) AS transaction_date,
      SUM(transaction_total_product_quantity) AS transaction_total_product_quantity,
      SUM(product_revenue) AS product_revenue,
      AVG(product_price) AS product_price
    FROM
      `{project_id}.{dataset_id}.transaction_product_level_quantity_staging_data`
    GROUP BY
      1,
      2) transaction_product_level_quantity_data
  ON
    customer_level_transaction_refund_data.transaction_id = transaction_product_level_quantity_data.transaction_id ),
  customer_product_level_transaction AS (
  SELECT
    EMAIL_HASH,
    productsku,
    transaction_date,
    COUNT(DISTINCT transaction_id) AS unique_transaction_count
  FROM
    transaction_customer_product_level_data
  GROUP BY
    1,
    2,
    3 ),
  customer_product_level_refund AS (
  SELECT
    transaction_to_email_hash_mapping.EMAIL_HASH AS EMAIL_HASH,
    NULLIF(PRODUCT_SKU, '') AS productsku,
    DATE(DATE) AS transaction_date,
    COUNT(DISTINCT web_analytics.TRANSACTION_ID) AS unique_refund_transaction_count
  FROM
    `{project_id}.{dataset_id}.transaction_to_email_hash_mapping` transaction_to_email_hash_mapping
  LEFT JOIN
    `{source_table_project_id}.{source_table_dataset_id}.{source_table_refund_data_table_name}` web_analytics
  ON
    CAST(web_analytics.TRANSACTION_ID AS STRING) = CAST(transaction_to_email_hash_mapping.transaction_id AS STRING)
  WHERE
    APLICATION_TYPE = 'Web Responsive'
    AND CAST(PRODUCT_REFUND_QUANTITY AS FLOAT64)<0
  GROUP BY 1,2,3 ),
  customer_product_past_transactions_refund_data AS (
  SELECT
    customer_product_level_transaction.EMAIL_HASH,
    customer_product_level_transaction.productsku,
    customer_product_level_transaction.transaction_date,
    customer_product_level_transaction.unique_transaction_count AS past_unique_transaction_count,
    IFNULL(unique_refund_transaction_count,0) AS past_unique_refund_transaction_count
  FROM
    customer_product_level_transaction
  LEFT JOIN
    customer_product_level_refund
  ON
    customer_product_level_transaction.productsku = customer_product_level_refund.productsku
    AND customer_product_level_transaction.transaction_date = customer_product_level_refund.transaction_date
    AND customer_product_level_transaction.EMAIL_HASH = customer_product_level_refund.EMAIL_HASH )
SELECT
  transaction_id,
  SUM(IFNULL(same_product_same_customer_past_unique_transaction_count,0)) AS same_product_same_customer_past_unique_transaction_count,
  SUM(CASE
      WHEN same_product_same_customer_past_unique_refund_transaction_count>same_product_same_customer_past_unique_transaction_count THEN same_product_same_customer_past_unique_transaction_count
    ELSE
    same_product_same_customer_past_unique_refund_transaction_count
  END
    ) AS same_product_same_customer_past_unique_refund_transaction_count,
  SUM(CASE
      WHEN same_product_same_customer_past_unique_refund_transaction_count>0 THEN current_transaction_product_revenue
    ELSE
    0
  END
    ) AS estimated_refund_value_based_on_customer_past_refund_history
FROM (
  SELECT
    current_transaction.transaction_id,
    current_transaction.productsku,
    current_transaction.EMAIL_HASH,
    MIN(current_transaction.transaction_total_product_quantity) AS current_transaction_product_quantity,
    MIN(current_transaction.product_revenue) AS current_transaction_product_revenue,
    SUM(IFNULL(historical_transaction.past_unique_transaction_count,0)) AS same_product_same_customer_past_unique_transaction_count,
    SUM(IFNULL(historical_transaction.past_unique_refund_transaction_count,0)) AS same_product_same_customer_past_unique_refund_transaction_count
  FROM
    transaction_customer_product_level_data current_transaction
  LEFT JOIN
    customer_product_past_transactions_refund_data historical_transaction
  ON
    current_transaction.productsku = historical_transaction.productsku
    AND current_transaction.EMAIL_HASH = historical_transaction.EMAIL_HASH
    AND current_transaction.transaction_date>historical_transaction.transaction_date
  WHERE
    current_transaction.EMAIL_HASH IS NOT NULL
  GROUP BY
    1,2,3 )
GROUP BY
  transaction_id;
