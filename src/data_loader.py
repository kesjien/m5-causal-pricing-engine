import pandas as pd
import os

def downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimizes memory usage by downcasting numeric columns."""
    float_cols = df.select_dtypes(include=['float']).columns
    int_cols = df.select_dtypes(include=['integer']).columns
    
    df[float_cols] = df[float_cols].astype('float32')
    df[int_cols] = df[int_cols].astype('int32')
    return df

def load_and_merge_data(data_dir: str) -> pd.DataFrame:
    """Loads M5 calendar, prices, and sales data, and merges them into a long format table."""
    sales_path = os.path.join(data_dir, "sales_train_validation.csv")
    calendar_path = os.path.join(data_dir, "calendar.csv")
    prices_path = os.path.join(data_dir, "sell_prices.csv")
    
    sales_df = pd.read_csv(sales_path)
    calendar_df = pd.read_csv(calendar_path)
    prices_df = pd.read_csv(prices_path)
    
    # Melt sales from wide to long format
    id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    sales_long = pd.melt(sales_df, id_vars=id_vars, var_name="d", value_name="sales")
    
    # Merge calendar and prices
    merged = sales_long.merge(calendar_df, on="d", how="left")
    merged = merged.merge(prices_df, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    
    return downcast_dtypes(merged)