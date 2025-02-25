def remove_duplicate_signs(df, lat_col, lon_col, decimals=5):
    df = df.copy()
    df["lat_round"] = df[lat_col].round(decimals)
    df["lon_round"] = df[lon_col].round(decimals)
    original_count = len(df)
    df_unique = df.drop_duplicates(subset=["lat_round", "lon_round"])
    duplicates_removed = original_count - len(df_unique)
    df_unique = df_unique.drop(columns=["lat_round", "lon_round"])
    return df_unique, duplicates_removed
