def get_all_station_metadata():
    """
    Loads and combines metadata for both ODA and SIAS stations.
    """
    try:
        oda_meta = pd.read_csv(ODA_STATIONS_PATH)
        oda_meta.rename(columns={'Station_ID': 'station_id', 'X': 'x_coord', 'Y': 'y_coord'}, inplace=True)
        
        sias_meta = pd.read_csv(SIAS_STATIONS_PATH, encoding='latin1')
        sias_meta.rename(columns={'id': 'station_id', 'Est_cord': 'x_coord', 'Nord_cord': 'y_coord'}, inplace=True)
        
        # Combine and keep only relevant columns
        all_meta = pd.concat([
            oda_meta[['station_id', 'x_coord', 'y_coord']],
            sias_meta[['station_id', 'x_coord', 'y_coord']]
        ], ignore_index=True).drop_duplicates(subset='station_id')
        
        return all_meta
    except FileNotFoundError as e:
        print(f"Error loading metadata file: {e}")
        return None

def plot_homogeneity_classification(results_gdf, sicily_gdf):
    """
    Plots all stations on a map, colored by their homogeneity class.
    """
    print("\n--- Generating classification map ---")
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    sicily_gdf.plot(ax=ax, color='lightgray', edgecolor='black')
    
    # Define colors for a clear visual distinction
    color_map = {
        'Useful': 'green',
        'Doubtful': 'orange',
        'Suspect': 'red'
    }
    
    # Plot each class separately to create a proper legend
    for classification, color in color_map.items():
        subset = results_gdf[results_gdf['classification'] == classification]
        if not subset.empty:
            subset.plot(ax=ax, marker='o', color=color, markersize=50, label=classification, alpha=0.8)
            
    ax.set_title('Homogeneity Classification of Rainfall Stations', fontsize=16)
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.legend(title='Classification', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('homogeneity_classification_map.png', dpi=300)
    print("Saved classification map to 'homogeneity_classification_map.png'")
    plt.show()

def plot_sias_auto_homogeneity(sias_results_gdf, sicily_gdf):
    """
    Plots SIAS stations with break points highlighted in two separate maps.
    """
    print("\n--- Generating SIAS auto-homogeneity map ---")
    
    # Convert date columns to datetime, coercing errors to NaT (Not a Time)
    sias_results_gdf['break_p1'] = pd.to_datetime(sias_results_gdf['break_found_period1 (->2009)'], errors='coerce')
    sias_results_gdf['break_p2'] = pd.to_datetime(sias_results_gdf['break_found_period2 (2009-2013)'], errors='coerce')

    fig, axes = plt.subplots(1, 2, figsize=(20, 10), sharex=True, sharey=True)
    fig.suptitle('SIAS Stations Auto-Homogeneity Test Results', fontsize=18)
    
    # --- Plot for Period 1 (-> 2009) ---
    sicily_gdf.plot(ax=axes[0], color='lightgray', edgecolor='black')
    sias_results_gdf.plot(ax=axes[0], marker='o', color='gray', markersize=30, label='No Break Detected')
    
    break_p1_stations = sias_results_gdf[sias_results_gdf['break_p1'].notna()]
    if not break_p1_stations.empty:
        break_p1_stations.plot(ax=axes[0], marker='X', color='red', markersize=80, label='Break Detected')
        
    axes[0].set_title('Break Points Detected (pre-2009)', fontsize=14)
    axes[0].set_xlabel('Easting (m)')
    axes[0].set_ylabel('Northing (m)')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # --- Plot for Period 2 (2009-2013) ---
    sicily_gdf.plot(ax=axes[1], color='lightgray', edgecolor='black')
    sias_results_gdf.plot(ax=axes[1], marker='o', color='gray', markersize=30, label='No Break Detected')

    break_p2_stations = sias_results_gdf[sias_results_gdf['break_p2'].notna()]
    if not break_p2_stations.empty:
        break_p2_stations.plot(ax=axes[1], marker='X', color='red', markersize=80, label='Break Detected')
        
    axes[1].set_title('Break Points Detected (2009-2013)', fontsize=14)
    axes[1].set_xlabel('Easting (m)')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
    plt.savefig('sias_auto_homogeneity_map.png', dpi=300)
    print("Saved SIAS map to 'sias_auto_homogeneity_map.png'")
    plt.show()

def main():
    """Main function to run the visualization."""
    try:
        # Load analysis results
        homogeneity_results = pd.read_csv('homogeneity_classification.csv')
        sias_auto_results = pd.read_csv('sias_auto_homogeneity_results.csv')

        # Load station metadata and Sicily shapefile
        all_meta = get_all_station_metadata()
        sicily = gpd.read_file(SICILY_SHAPEFILE_PATH)

        if all_meta is None:
            return

        # --- FIX APPLIED HERE: CRS Transformation ---

        # 1. Prepare data for the classification plot
        merged_results = pd.merge(homogeneity_results, all_meta, on='station_id')
        results_gdf = gpd.GeoDataFrame(
            merged_results, geometry=gpd.points_from_xy(merged_results.x_coord, merged_results.y_coord)
        )
        # Define the initial CRS of the station points (likely UTM 33N)
        results_gdf.set_crs(epsg=32633, inplace=True)
        # REPROJECT the points to match the shapefile's CRS
        results_gdf = results_gdf.to_crs(sicily.crs)
        
        # 2. Prepare data for the SIAS auto-homogeneity plot
        merged_sias_results = pd.merge(sias_auto_results, all_meta, on='station_id')
        sias_results_gdf = gpd.GeoDataFrame(
            merged_sias_results, geometry=gpd.points_from_xy(merged_sias_results.x_coord, merged_sias_results.y_coord)
        )
        # Define the initial CRS of the station points
        sias_results_gdf.set_crs(epsg=32633, inplace=True)
        # REPROJECT the points to match the shapefile's CRS
        sias_results_gdf = sias_results_gdf.to_crs(sicily.crs)
        
        # --- End of FIX ---

        # Generate plots with the correctly aligned data
        plot_homogeneity_classification(results_gdf, sicily)
        plot_sias_auto_homogeneity(sias_results_gdf, sicily)

    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure that 'homogeneity_classification.csv', 'sias_auto_homogeneity_results.csv', and all metadata/shapefiles are present.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()


"""
This section finds high-quality pairs of ODA and SIAS stations based on
proximity, data coverage, and homogeneity, and generates scatter plots
to visually compare their rainfall records.
"""

HOMOGENEITY_RESULTS_PATH = 'homogeneity_classification.csv'
DISTANCE_THRESHOLD_METERS = 10000  # 10 km
COVERAGE_THRESHOLD_PERCENT = 10.0
WINDOW_DURATION_YEARS = 10
OVERALL_SEARCH_START = pd.to_datetime('2002-01-01')
OVERALL_SEARCH_END = pd.to_datetime('2021-12-02')
NUM_PAIRS_TO_PLOT =2

def load_and_prepare_sias_data():
    """
    Loads SIAS station metadata and rainfall data from the specified Pickle file.
    This version is robust and does not assume column names, only their order.
    """
    print("\n--- Loading and Preparing SIAS Dataset from Pickle File ---")
    try:
        stations_df = pd.read_csv(SIAS_STATIONS_PATH, encoding='latin1')
        stations_df.rename(columns={'id': 'station_id', 'Est_cord': 'x_coord', 'Nord_cord': 'y_coord'}, inplace=True)
    except FileNotFoundError as e:
        print(f"Error loading SIAS station file: {e}")
        return None

    try:
        print(f"Reading SIAS data from pickle file: {SIAS_DATA_PATH}")
        with open(SIAS_DATA_PATH, 'rb') as f:
            sias_pickle_data = pickle.load(f)

        all_dfs = []
        # --- NEW ROBUST LOGIC ---
        # Get the first valid dataframe to determine column names
        first_key = next(iter(sias_pickle_data))
        sample_df = sias_pickle_data[first_key]
        
        # Assume the first column is ID and the second is rain
        id_col_name = sample_df.columns[0]
        rain_col_name = sample_df.columns[1]
        print(f"Inferred SIAS columns -> ID: '{id_col_name}', Rain: '{rain_col_name}'")

        for date_str, daily_df in sias_pickle_data.items():
            if not daily_df.empty:
                # Rename columns based on position
                renamed_df = daily_df.rename(columns={
                    id_col_name: 'station_id',
                    rain_col_name: 'rain'
                })
                renamed_df['date'] = pd.to_datetime(date_str)
                all_dfs.append(renamed_df)
        
        long_df = pd.concat(all_dfs, ignore_index=True)

    except FileNotFoundError:
        print(f"Error: The pickle file was not found at '{SIAS_DATA_PATH}'.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the pickle file: {e}")
        return None

    final_df = pd.merge(long_df, stations_df[['station_id', 'x_coord', 'y_coord']], on='station_id')
    final_df['data_source'] = 'sias'
    final_df['date'] = pd.to_datetime(final_df['date'])
    final_df = final_df[
        (final_df['date'].dt.year >= START_YEAR_SIAS) & 
        (final_df['date'].dt.year <= END_YEAR)
    ]

    print(f"SIAS data processed from pickle file. Shape: {final_df.shape}")
    return final_df[['date', 'station_id', 'rain', 'x_coord', 'y_coord', 'data_source']]


def get_combined_data():
    """
    Loads, prepares, validates, and combines ODA and SIAS data.
    """
    oda_df = load_oda_data()
    sias_df = load_and_prepare_sias_data()

    if oda_df is None or sias_df is None:
        return None, None

    # --- NEW VALIDATION STEP ---
    required_cols = ['station_id', 'date', 'rain']
    if not all(col in oda_df.columns for col in required_cols):
        print("Error: ODA DataFrame is missing required columns. Please check load_oda_data.")
        return None, None
    if not all(col in sias_df.columns for col in required_cols):
        print("Error: SIAS DataFrame is missing required columns. Please check load_and_prepare_sias_data.")
        return None, None

    # Combine the two data sources
    combined_df = pd.concat([oda_df, sias_df], ignore_index=True)
    combined_df['date'] = pd.to_datetime(combined_df['date'])

    # Create a unified station metadata table
    station_meta = combined_df[['station_id', 'x_coord', 'y_coord', 'data_source']].drop_duplicates(subset=['station_id']).reset_index(drop=True)

    return combined_df, station_meta


def find_best_10yr_window(oda_id, sias_id, all_data):
    """
    Slides a 10-year window backwards in time to find the most recent period
    where both stations meet the data coverage threshold.

    Args:
        oda_id (int): The ID of the ODA station.
        sias_id (int): The ID of the SIAS station.
        all_data (pd.DataFrame): The combined dataframe of all rainfall data.

    Returns:
        tuple: (start_date, end_date) if a valid window is found, otherwise (None, None).
    """
    oda_series = all_data[all_data['station_id'] == oda_id][['date', 'rain']].set_index('date')
    sias_series = all_data[all_data['station_id'] == sias_id][['date', 'rain']].set_index('date')
    
    window_delta = pd.DateOffset(years=WINDOW_DURATION_YEARS)
    
    # Start searching from the end of the overall period
    current_end_date = OVERALL_SEARCH_END
    
    while (current_end_date - window_delta) >= OVERALL_SEARCH_START:
        current_start_date = current_end_date - window_delta
        
        # Check coverage for ODA station in the current window
        oda_subset = oda_series[(oda_series.index >= current_start_date) & (oda_series.index <= current_end_date)]
        
        # Check coverage for SIAS station
        sias_subset = sias_series[(sias_series.index >= current_start_date) & (sias_series.index <= current_end_date)]
        
        total_days_in_window = (current_end_date - current_start_date).days + 1
        
        oda_coverage = (oda_subset['rain'].notna().sum() / total_days_in_window) * 100
        sias_coverage = (sias_subset['rain'].notna().sum() / total_days_in_window) * 100
        
        if oda_coverage >= COVERAGE_THRESHOLD_PERCENT and sias_coverage >= COVERAGE_THRESHOLD_PERCENT:
            return current_start_date, current_end_date # Found the best (most recent) window
            
        # Move the window one month back to speed up the search
        current_end_date -= pd.DateOffset(months=1)
        
    return None, None # No suitable window found


def find_station_pairs(all_data, station_meta, homogeneity_results):
    """
    Finds the best-matched station pairs by dynamically searching for optimal time windows.
    """
    print("\n--- Finding best matched ODA-SIAS station pairs (Dynamic Window) ---")
    
    useful_station_ids = homogeneity_results[homogeneity_results['classification'] == 'Useful']['station_id']
    useful_station_meta = station_meta[station_meta['station_id'].isin(useful_station_ids)]
    
    meta_gdf = gpd.GeoDataFrame(
        useful_station_meta, 
        geometry=gpd.points_from_xy(useful_station_meta.x_coord, useful_station_meta.y_coord),
        crs="EPSG:32633"
    )
    
    oda_gdf = meta_gdf[meta_gdf['data_source'] == 'oda'].copy()
    sias_gdf = meta_gdf[meta_gdf['data_source'] == 'sias'].copy()
    
    print(f"Found {len(oda_gdf)} useful ODA stations and {len(sias_gdf)} useful SIAS stations.")
    
    if oda_gdf.empty or sias_gdf.empty:
        print("Could not find useful stations in one or both datasets.")
        return None
        
    matched_pairs = []
    
    # Use tqdm for a progress bar as this process can be slow
    for _, oda_station in tqdm(oda_gdf.iterrows(), total=oda_gdf.shape[0], desc="Finding Pairs"):
        buffer = oda_station.geometry.buffer(DISTANCE_THRESHOLD_METERS)
        candidate_sias = sias_gdf[sias_gdf.within(buffer)]
        
        if candidate_sias.empty:
            continue           
def find_station_pairs(all_data, station_meta, homogeneity_results):
    """
    Finds the best-matched station pairs by dynamically searching for optimal time windows.
    """
    print("\n--- Finding best matched ODA-SIAS station pairs (Dynamic Window) ---")
    
    useful_station_ids = homogeneity_results[homogeneity_results['classification'] == 'Useful']['station_id']
    useful_station_meta = station_meta[station_meta['station_id'].isin(useful_station_ids)]
    
    meta_gdf = gpd.GeoDataFrame(
        useful_station_meta, 
        geometry=gpd.points_from_xy(useful_station_meta.x_coord, useful_station_meta.y_coord),
        crs="EPSG:32633"
    )
    
    oda_gdf = meta_gdf[meta_gdf['data_source'] == 'oda'].copy()
    sias_gdf = meta_gdf[meta_gdf['data_source'] == 'sias'].copy()
    
    print(f"Found {len(oda_gdf)} useful ODA stations and {len(sias_gdf)} useful SIAS stations.")
    
    if oda_gdf.empty or sias_gdf.empty:
        print("Could not find useful stations in one or both datasets.")
        return None
        
    matched_pairs = []
    
    # Use tqdm for a progress bar as this process can be slow
    for _, oda_station in tqdm(oda_gdf.iterrows(), total=oda_gdf.shape[0], desc="Finding Pairs"):
        buffer = oda_station.geometry.buffer(DISTANCE_THRESHOLD_METERS)
        candidate_sias = sias_gdf[sias_gdf.within(buffer)]
        
        if candidate_sias.empty:
            continue
            
        potential_pairs = []
        for _, sias_station in candidate_sias.iterrows():
            start_date, end_date = find_best_10yr_window(oda_station['station_id'], sias_station['station_id'], all_data)
            
            if start_date is not None:
                distance = oda_station.geometry.distance(sias_station.geometry)
                potential_pairs.append({
                    'oda_id': oda_station['station_id'],
                    'sias_id': sias_station['station_id'],
                    'distance_m': distance,
                    'oda_x': oda_station.x_coord,
                    'oda_y': oda_station.y_coord,
                    'window_start': start_date,
                    'window_end': end_date
                })
                
        if potential_pairs:
            best_pair = min(potential_pairs, key=lambda x: x['distance_m'])
            matched_pairs.append(best_pair)
            
    if not matched_pairs:
        print("No matched pairs found meeting all criteria.")
        return None
        
    pairs_df = pd.DataFrame(matched_pairs).drop_duplicates(subset=['sias_id'])
    print(f"Found {len(pairs_df)} high-quality pairs with valid 10-year windows.")
    return pairs_df


def plot_rainfall_comparison(pairs_to_plot, all_data):
    """
    Generates scatter plots for pairs, using the specific time window found for each.
    """
    print("\n--- Generating rainfall comparison plots ---")
    fig, axes = plt.subplots(3, 2, figsize=(15, 20), constrained_layout=True)
    axes = axes.flatten()
    
    for i, (_, pair) in enumerate(pairs_to_plot.iterrows()):
        ax = axes[i]
        oda_id = pair['oda_id']
        sias_id = pair['sias_id']
        start = pair['window_start']
        end = pair['window_end']

        date_filter = (all_data['date'] >= start) & (all_data['date'] <= end)
        oda_series = all_data[(all_data['station_id'] == oda_id) & date_filter][['date', 'rain']].rename(columns={'rain': 'rain_oda'})
        sias_series = all_data[(all_data['station_id'] == sias_id) & date_filter][['date', 'rain']].rename(columns={'rain': 'rain_sias'})
        
        merged = pd.merge(oda_series, sias_series, on='date').dropna()
        x, y = merged['rain_oda'], merged['rain_sias']
        
        ax.scatter(x, y, alpha=0.4, s=15, edgecolors='k', linewidths=0.5)
        lims = [0, max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, 'r--', label='1:1 Line', zorder=5)
        
        correlation, _ = stats.pearsonr(x, y)
        rmse = np.sqrt(mean_squared_error(x, y))
        
        stats_text = f"Correlation: {correlation:.3f}\nRMSE: {rmse:.2f} mm"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.8))
        
        # Updated title to show the dynamic window
        title = (f"ODA ({oda_id}) vs SIAS ({sias_id})\n"
                 f"Period: {start.year}-{end.year}")
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("ODA Daily Rainfall (mm)", fontsize=12)
        ax.set_ylabel("SIAS Daily Rainfall (mm)", fontsize=12)
        ax.set_aspect('equal', 'box')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

    for j in range(len(pairs_to_plot), len(axes)):
        axes[j].set_visible(False)
        
    fig.suptitle("Direct Rainfall Comparison for Matched Station Pairs (Dynamic 10-Year Window)", fontsize=20)
    plt.savefig('rainfall_comparison_scatter_plots_dynamic.png', dpi=300, bbox_inches='tight')
    print("Saved comparison plot to 'rainfall_comparison_scatter_plots_dynamic.png'")
    plt.show()

def main():
    """Main function to orchestrate the station comparison."""
    data_df, station_meta = get_combined_data()
    
    try:
        homogeneity_results = pd.read_csv(HOMOGENEITY_RESULTS_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find '{HOMOGENEITY_RESULTS_PATH}'. Please run the main analysis script first.")
        return

    if data_df is None:
        print("Data loading failed. Exiting.")
        return
        
    all_pairs = find_station_pairs(data_df, station_meta, homogeneity_results)
    
    if all_pairs is None or len(all_pairs) < NUM_PAIRS_TO_PLOT:
        print(f"\nCould not find enough pairs ({NUM_PAIRS_TO_PLOT} required) to generate the plot.")
        if all_pairs is not None:
            print(f"Found only {len(all_pairs)} pairs.")
        return
        
    all_pairs_sorted = all_pairs.sort_values('oda_y', ascending=False)
    indices = np.linspace(0, len(all_pairs_sorted) - 1, NUM_PAIRS_TO_PLOT, dtype=int)
    pairs_to_plot = all_pairs_sorted.iloc[indices]
    
    print("\nSelected the following 5 pairs for plotting:")
    print(pairs_to_plot[['oda_id', 'sias_id', 'distance_m', 'window_start', 'window_end']].round(2))
    
    plot_rainfall_comparison(pairs_to_plot, data_df)

if __name__ == '__main__':
    main()




