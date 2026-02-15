def create_master_feature_set(df):
    logging.info("Engineering the V3 master feature set...")
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    df.dropna(subset=['Order Date'], inplace=True)

    if 'Profit_Margin' not in df.columns:
        df['Profit_Margin'] = np.where(df['Total Sales'] > 0, df['Total Profit'] / df['Total Sales'], 0)

    df['month_of_year'] = df['Order Date'].dt.month
    df['first_order_date'] = df.groupby('Customer ID')['Order Date'].transform('min')
    df['days_since_first_order'] = (df['Order Date'] - df['first_order_date']).dt.days

    df['postal_code_profitability'] = df.groupby('Postal Code')['Profit_Margin'].transform('mean')
    df['state_sales_volume'] = df.groupby('State')['Total Sales'].transform('sum')
    df['subcategory_avg_margin'] = df.groupby('Sub-Category')['Profit_Margin'].transform('mean')
    df['category_avg_discount'] = df.groupby('Category')['Discount'].transform('mean')
    df['customer_avg_order_size'] = df.groupby('Customer ID')['Total Sales'].transform('mean')
    
    df.fillna(df.median(numeric_only=True), inplace=True)
    return df