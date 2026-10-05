-- ShelfCast: cleaning and daily sales. Runs unchanged on Snowflake and on DuckDB.
-- Input: RAW.TRANSACTIONS (one row per invoice line, loaded by `python -m shelfcast load`).

CREATE SCHEMA IF NOT EXISTS MART;

-- Real sales only: no cancellations, returns, postage, fees or manual adjustments.
-- Product codes start with five digits (85123A, 22423, ...); service codes such as POST,
-- DOT, M, BANK CHARGES or gift vouchers do not. Large orders (1,000+ units) that the same
-- customer later cancelled are dropped too, so one-off mistakes do not look like demand.
CREATE OR REPLACE VIEW MART.CLEAN_SALES AS
WITH big_cancellations AS (
    SELECT DISTINCT customer_id, stock_code, -quantity AS quantity
    FROM RAW.TRANSACTIONS
    WHERE invoice LIKE 'C%' AND quantity <= -1000
)
SELECT
    t.invoice,
    t.stock_code,
    t.description,
    t.quantity,
    t.price,
    t.customer_id,
    t.country,
    t.invoice_date,
    CAST(t.invoice_date AS DATE) AS sale_date
FROM RAW.TRANSACTIONS t
LEFT JOIN big_cancellations c
    ON c.customer_id = t.customer_id
   AND c.stock_code = t.stock_code
   AND c.quantity = t.quantity
WHERE c.stock_code IS NULL
  AND t.quantity > 0
  AND t.price > 0
  AND t.invoice NOT LIKE 'C%'
  AND TRY_CAST(SUBSTR(t.stock_code, 1, 5) AS INTEGER) IS NOT NULL;

-- Units sold per product per day (days without sales are simply missing here).
CREATE OR REPLACE TABLE MART.DAILY_SALES AS
SELECT
    stock_code,
    sale_date,
    SUM(quantity) AS units,
    SUM(quantity * price) AS revenue,
    COUNT(DISTINCT invoice) AS orders
FROM MART.CLEAN_SALES
GROUP BY stock_code, sale_date;

-- One row per product, used to pick which products to forecast.
CREATE OR REPLACE TABLE MART.PRODUCT_SUMMARY AS
SELECT
    stock_code,
    MAX(description) AS description,
    SUM(quantity) AS total_units,
    SUM(quantity * price) AS total_revenue,
    COUNT(DISTINCT sale_date) AS selling_days,
    MIN(sale_date) AS first_sale,
    MAX(sale_date) AS last_sale,
    MEDIAN(price) AS median_price
FROM MART.CLEAN_SALES
GROUP BY stock_code;
