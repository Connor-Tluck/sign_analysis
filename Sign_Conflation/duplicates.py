def remove_duplicate_signs(df, lat_col, lon_col, decimals=5):
    df = df.copy()
    # Round coordinates to approximate a buffer distance
    df["lat_round"] = df[lat_col].round(decimals)
    df["lon_round"] = df[lon_col].round(decimals)
    original_count = len(df)

    # If there is a MUTCD column with meaningful data, drop duplicates based on lat, lon, and MUTCD.
    if "MUTCD" in df.columns:
        # Check if all MUTCD values are "UNKNOWN" (or empty after stripping); if so, don't remove duplicates.
        if not df["MUTCD"].astype(str).str.strip().str.upper().eq("UNKNOWN").all():
            df_unique = df.drop_duplicates(subset=["lat_round", "lon_round", "MUTCD"])
        else:
            df_unique = df.copy()
    else:
        df_unique = df.copy()

    duplicates_removed = original_count - len(df_unique)
    df_unique = df_unique.drop(columns=["lat_round", "lon_round"])
    return df_unique, duplicates_removed
