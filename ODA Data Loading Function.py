# -*- coding: utf-8 -*-
"""
DATALOADER file for the rainfall data homogeneity project.

@author: Niloufar Beikahmadi
"""

def load_oda_data():
    """
    Loads and prepares the ODA rainfall data.
    """
    print("--- Loading and Preparing ODA Dataset ---")
    try:
        stations_df = pd.read_csv(ODA_STATIONS_PATH)
        stations_df.rename(columns={'Station_ID': 'station_id', 'X': 'x_coord', 'Y': 'y_coord'}, inplace=True)
        
        data_df = pd.read_csv(ODA_DATA_PATH)
        data_df.rename(columns={'Year_Date': 'date'}, inplace=True)
        
        # --- FIX APPLIED HERE ---
        # Removed the explicit format to allow pandas to infer it automatically.
        data_df['date'] = pd.to_datetime(data_df['date'])
        
        # Filter data for the specified period
        data_df = data_df[(data_df['date'].dt.year >= START_YEAR_ODA) & (data_df['date'].dt.year <= END_YEAR)]
        
        # Melt the dataframe to long format
        long_df = data_df.melt(id_vars=['date'], var_name='station_code', value_name='rain')
        
        # The station codes in the data file are like 'V1', 'V2', etc. We need to extract the number.
        long_df['station_id'] = long_df['station_code'].str.replace('V', '').astype(int)
        
        # Merge with station metadata
        final_df = pd.merge(long_df, stations_df[['station_id', 'x_coord', 'y_coord']], on='station_id')
        final_df['data_source'] = 'oda'
        
        print(f"ODA data processed. Shape: {final_df.shape}")
        return final_df[['date', 'station_id', 'rain', 'x_coord', 'y_coord', 'data_source']]

    except FileNotFoundError as e:
        print(f"Error loading ODA file: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred in load_oda_data: {e}")
        return None
