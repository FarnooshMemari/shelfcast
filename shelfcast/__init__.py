"""ShelfCast: demand forecasts that turn into restock decisions.

Snowflake holds the data and the business logic, a PyTorch model forecasts demand with
uncertainty, and a Streamlit app turns the forecasts into an order list.
"""

__version__ = "0.1.0"
