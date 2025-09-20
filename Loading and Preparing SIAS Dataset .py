import pickle

SIAS_DATA_PATH = 'sias.pickle' 

def load_and_prepare_sias_data():
    """
    Loads the SIAS station metadata and rainfall data from the specified Pickle file,
    and transforms it into a standardized long format.
    """
    print("\n--- Loading and Preparing SIAS Dataset from Pickle File ---")
    try:
        # Load station metadata first
        stations_df = pd.read_csv(SIAS_STATIONS_PATH, encoding='latin1')
        stations_df.rename(columns={'id': 'station_id', 'Est_cord': 'x_coord', 'Nord_cord': 'y_coord'}, inplace=True)
    except FileNotFoundError as e:
        print(f"Error loading SIAS station file: {e}")
        return None

    try:
        # --- NEW LOGIC FOR READING THE PICKLE FILE ---

        # 1. Load the entire pickle file into a dictionary
        print(f"Reading SIAS data from pickle file: {SIAS_DATA_PATH}")
        with open(SIAS_DATA_PATH, 'rb') as f:
            sias_pickle_data = pickle.load(f)

        # 2. Iterate through the dictionary and build a list of DataFrames
        all_dfs = []
        for date_str, daily_df in sias_pickle_data.items():
            # Add the date (from the dictionary key) as a new column
            daily_df['date'] = pd.to_datetime(date_str)
            all_dfs.append(daily_df)

        # 3. Concatenate all the daily DataFrames into one large DataFrame
        long_df = pd.concat(all_dfs, ignore_index=True)

        # --- IMPORTANT ASSUMPTION ---
        # Based on your initial code, we assume the columns in the daily
        # DataFrames are named 'id' and 'rain'. If they are different,
        # you will need to adjust the line below.
        long_df.rename(columns={'id': 'station_id'}, inplace=True)

        # --- END OF NEW LOGIC ---

    except FileNotFoundError:
        print(f"Error: The pickle file was not found at '{SIAS_DATA_PATH}'.")
        print("Please ensure the path in config.py is correct.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the pickle file: {e}")
        return None

    # Merge with station metadata to get coordinates
    final_df = pd.merge(long_df, stations_df[['station_id', 'x_coord', 'y_coord']], on='station_id')

    # Add data source identifier and filter by date
    final_df['data_source'] = 'sias'
    final_df['date'] = pd.to_datetime(final_df['date'])
    final_df = final_df[
        (final_df['date'].dt.year >= START_YEAR_SIAS) & 
        (final_df['date'].dt.year <= END_YEAR)
    ]

    print(f"SIAS data processed from pickle file. Shape: {final_df.shape}")
    return final_df[['date', 'station_id', 'rain', 'x_coord', 'y_coord', 'data_source']]
