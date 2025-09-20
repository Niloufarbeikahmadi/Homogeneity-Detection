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



