-- ShelfCast: one-time Snowflake setup (Snowflake only, not needed for the local DuckDB mode).
--
-- How to run it:
--   1. Sign in to Snowsight (a free trial account is fine).
--   2. Open Projects > Workspaces (or Worksheets) and create a new SQL file.
--   3. Paste this whole file and replace every YOUR_USER_NAME with your Snowflake user name
--      (run  SELECT CURRENT_USER();  first if you are not sure what it is).
--   4. Choose "Run All". The result of the LAST statement shows token_secret. Copy it into
--      .env as SNOWFLAKE_PAT right away, because Snowflake shows it only once.
--   5. Run the commented query at the very bottom on its own to get SNOWFLAKE_ACCOUNT.

USE ROLE ACCOUNTADMIN;

-- 1) A small warehouse that switches itself off after 60 idle seconds, to save credits.
CREATE WAREHOUSE IF NOT EXISTS SHELFCAST_WH
  WAREHOUSE_SIZE = 'XSMALL'
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE
  INITIALLY_SUSPENDED = TRUE;

-- 2) A role that can only touch this project.
CREATE ROLE IF NOT EXISTS SHELFCAST_ROLE;
GRANT ROLE SHELFCAST_ROLE TO ROLE SYSADMIN;
GRANT ROLE SHELFCAST_ROLE TO USER YOUR_USER_NAME;
GRANT USAGE, OPERATE ON WAREHOUSE SHELFCAST_WH TO ROLE SHELFCAST_ROLE;
GRANT DATABASE ROLE SNOWFLAKE.CORTEX_USER TO ROLE SHELFCAST_ROLE;  -- for the AI brief

-- 3) The database. The project role owns it, so the pipeline can create its own tables.
CREATE DATABASE IF NOT EXISTS SHELFCAST;
GRANT OWNERSHIP ON DATABASE SHELFCAST TO ROLE SHELFCAST_ROLE COPY CURRENT GRANTS;
USE ROLE SHELFCAST_ROLE;
CREATE SCHEMA IF NOT EXISTS SHELFCAST.RAW;
CREATE SCHEMA IF NOT EXISTS SHELFCAST.MART;
USE ROLE ACCOUNTADMIN;

-- 4) Let Python sign in with a programmatic access token (PAT). By default Snowflake only
--    accepts PATs from users who have a network policy; this policy drops that requirement
--    for your user (a network policy is still enforced if you add one later).
CREATE SCHEMA IF NOT EXISTS SHELFCAST.ADMIN;
CREATE AUTHENTICATION POLICY IF NOT EXISTS SHELFCAST.ADMIN.SHELFCAST_PAT_POLICY
  PAT_POLICY = (NETWORK_POLICY_EVALUATION = ENFORCED_NOT_REQUIRED);
ALTER USER YOUR_USER_NAME SET AUTHENTICATION POLICY SHELFCAST.ADMIN.SHELFCAST_PAT_POLICY;

-- Optional: only if the AI brief later says the model is not available in your region.
-- ALTER ACCOUNT SET CORTEX_ENABLED_CROSS_REGION = 'ANY_REGION';

-- 5) The token itself: limited to the project role and valid for 30 days.
--    Keep this as the last statement so its result (token_secret) is what you see.
ALTER USER YOUR_USER_NAME ADD PROGRAMMATIC ACCESS TOKEN SHELFCAST_APP
  ROLE_RESTRICTION = 'SHELFCAST_ROLE'
  DAYS_TO_EXPIRY = 30
  COMMENT = 'ShelfCast pipeline and Streamlit app';

-- Run this line on its own: it prints the value for SNOWFLAKE_ACCOUNT (ORGNAME-ACCOUNTNAME).
-- SELECT CURRENT_ORGANIZATION_NAME() || '-' || CURRENT_ACCOUNT_NAME() AS snowflake_account;
