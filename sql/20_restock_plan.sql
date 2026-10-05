-- ShelfCast: the default restock plan as a view, so anyone can read it straight from the
-- warehouse (a Snowsight dashboard, a BI tool, another team's SQL).
-- Same rules as shelfcast/restock.py with the default settings: orders are placed every
-- 7 days and we plan for the P90 of total demand until the next delivery. The Streamlit
-- app recomputes the plan live when you move its sliders. Runs on Snowflake and DuckDB.
--
-- Inputs: MART.FORECASTS  one row per product and forecast day h = 1..14, with the running
--                         total forecast total_p10 / total_p50 / total_p90 (units from day 1 to h)
--         MART.INVENTORY  stock on hand, stock on order, supplier lead time, unit cost

CREATE OR REPLACE VIEW MART.RESTOCK_PLAN AS
WITH hz AS (
    SELECT MAX(h) AS max_h FROM MART.FORECASTS
),
stock AS (
    -- an order has to last until the next one arrives: lead time + 7 days between orders
    SELECT
        i.stock_code,
        i.on_hand,
        i.on_order,
        i.lead_time_days,
        i.unit_cost,
        LEAST(i.lead_time_days + 7, hz.max_h) AS cover_days,
        hz.max_h
    FROM MART.INVENTORY i
    CROSS JOIN hz
),
need AS (
    SELECT
        s.stock_code,
        f.total_p10 AS demand_low,
        f.total_p50 AS demand_mid,
        f.total_p90 AS demand_high
    FROM stock s
    JOIN MART.FORECASTS f ON f.stock_code = s.stock_code AND f.h = s.cover_days
),
runout AS (
    -- the first forecast day on which expected sales pass the stock on hand
    SELECT s.stock_code, MIN(f.h) - 1 AS days_left
    FROM stock s
    JOIN MART.FORECASTS f ON f.stock_code = s.stock_code AND f.total_p50 > s.on_hand
    GROUP BY s.stock_code
),
calc AS (
    SELECT
        s.stock_code,
        s.on_hand,
        s.lead_time_days,
        s.unit_cost,
        COALESCE(r.days_left, s.max_h) AS days_left,
        n.demand_low,
        n.demand_mid,
        n.demand_high,
        CAST(GREATEST(0, CEIL(n.demand_high - s.on_hand - s.on_order)) AS INTEGER) AS order_qty
    FROM stock s
    JOIN need n ON n.stock_code = s.stock_code
    LEFT JOIN runout r ON r.stock_code = s.stock_code
)
SELECT
    CASE
        WHEN calc.days_left < calc.lead_time_days THEN 'Order now'
        WHEN calc.order_qty > 0 THEN 'Order this week'
        ELSE 'OK'
    END AS status,
    calc.stock_code,
    p.description,
    calc.on_hand,
    calc.days_left,
    calc.lead_time_days,
    calc.demand_low,
    calc.demand_mid,
    calc.demand_high,
    calc.order_qty,
    calc.order_qty * calc.unit_cost AS order_value
FROM calc
LEFT JOIN MART.PRODUCT_SUMMARY p ON p.stock_code = calc.stock_code;
