def standard_num_vals(df):
    numeric_cols = df.select_dtypes(include=np.number)
    for col in numeric_cols.columns:
        valid = validate(df[col], (int, float))
        # if valid.isinstance(Exception):
        #     df[col].to_numeric()
    return numeric_cols