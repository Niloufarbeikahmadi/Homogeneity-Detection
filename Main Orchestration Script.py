# -*- coding: utf-8 -*-
"""
Main script to run the rainfall data homogeneity analysis.

@author: Niloufar Beikahmadi
"""
from data_loader import get_combined_data, filter_stations_by_shapefile
from analysis import find_neighbors, run_homogeneity_analysis, run_sias_auto_homogeneity

def main():
    """
    Main function to orchestrate the analysis.
    """
    print("Starting Rainfall Data Homogeneity Analysis")
    
    # 1. Load and combine data
    data_df, station_meta = get_combined_data()
    
    if data_df is None:
        print("Data loading failed. Exiting.")
        return
        
    # 2. Filter stations by Sicily shapefile
    station_meta_filtered = filter_stations_by_shapefile(station_meta)
    filtered_station_ids = station_meta_filtered['station_id'].unique()
    data_df_filtered = data_df[filtered_station_ids]
    
    # 3. Find neighbors
    neighbors = find_neighbors(station_meta_filtered)
    
    # 4. Run relative homogeneity analysis
    homogeneity_results = run_homogeneity_analysis(data_df_filtered, station_meta_filtered, neighbors)
    
    print("\n--- Homogeneity Analysis Results ---")
    print(homogeneity_results)
    homogeneity_results.to_csv('homogeneity_classification.csv', index=False)
    print("\nResults saved to homogeneity_classification.csv")
    
    # 5. Run SIAS auto-homogeneity analysis
    sias_auto_results = run_sias_auto_homogeneity(data_df_filtered, station_meta_filtered)
    
    print("\n--- SIAS Auto-homogeneity Analysis Results ---")
    print(sias_auto_results)
    sias_auto_results.to_csv('sias_auto_homogeneity_results.csv', index=False)
    print("\nSIAS auto-homogeneity results saved to sias_auto_homogeneity_results.csv")

if __name__ == '__main__':
    main()