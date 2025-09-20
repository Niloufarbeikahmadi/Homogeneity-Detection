# -*- coding: utf-8 -*-
"""
Core analysis script for homogeneity testing and station classification.

@author: Niloufar Beikahmadi
"""
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from homogeneity_tests import pettitt_test, snht_test
from config import NUM_NEIGHBORS, ALPHA, SNHT_WINDOW, SIAS_PERIOD_1_END, SIAS_PERIOD_2_END

def find_neighbors(station_meta):
    """
    Finds the nearest neighbors for each station.
    """
    print("\n--- Finding nearest neighbors for each station ---")
    coords = station_meta[['x_coord', 'y_coord']].values
    tree = cKDTree(coords)
    
    neighbors = {}
    for i, row in station_meta.iterrows():
        station_id = row['station_id']
        station_coords = [row['x_coord'], row['y_coord']]
        
        # Query for NUM_NEIGHBORS + 1 because the station itself will be the closest
        distances, indices = tree.query(station_coords, k=NUM_NEIGHBORS + 1)
        
        # Exclude the station itself
        neighbor_ids = station_meta.iloc[indices[1:]]['station_id'].tolist()
        neighbors[station_id] = neighbor_ids
        
    return neighbors

def run_homogeneity_analysis(data_df, station_meta, neighbors):
    """
    Runs the full homogeneity analysis workflow.
    """
    print("\n--- Running homogeneity analysis ---")
    results = []
    
    for station_id in data_df.columns:
        print(f"Analyzing station: {station_id}")
        candidate_series = data_df[station_id].dropna()
        
        if candidate_series.empty:
            continue
            
        neighbor_list = neighbors.get(station_id, [])
        
        pettitt_rejects = 0
        snht_rejects = 0
        
        for neighbor_id in neighbor_list:
            neighbor_series = data_df[neighbor_id].dropna()
            
            # Align series by date
            common_index = candidate_series.index.intersection(neighbor_series.index)
            if len(common_index) < 20: # Need sufficient overlap
                continue
            
            aligned_candidate = candidate_series.loc[common_index]
            aligned_neighbor = neighbor_series.loc[common_index]
            
            # Form difference series
            diff_series = (aligned_candidate - aligned_neighbor).values
            
            # Apply Pettitt test
            _, p_value = pettitt_test(diff_series)
            if p_value < ALPHA:
                pettitt_rejects += 1
                
            # Apply SNHT
            # We apply it to the whole series for simplicity, but a moving window approach is also valid
            max_t, _ = snht_test(diff_series, len(diff_series))
            # A simplified critical value check for SNHT
            if max_t > 11: # Critical value for alpha=0.05 is often around this value
                snht_rejects += 1
        
        total_rejects = pettitt_rejects + snht_rejects
        
        # Classification
        if total_rejects <= 1:
            classification = 'Useful'
        elif total_rejects == 2:
            classification = 'Doubtful'
        else:
            classification = 'Suspect'
            
        results.append({
            'station_id': station_id,
            'pettitt_rejects': pettitt_rejects,
            'snht_rejects': snht_rejects,
            'total_rejects': total_rejects,
            'classification': classification
        })
        
    return pd.DataFrame(results)

def run_sias_auto_homogeneity(data_df, station_meta):
    """
    Performs an auto-homogeneity test on SIAS stations.
    """
    print("\n--- Running auto-homogeneity analysis for SIAS stations ---")
    sias_stations = station_meta[station_meta['data_source'] == 'sias']['station_id']
    sias_data = data_df[sias_stations]
    
    results = []
    for station_id in sias_data.columns:
        series = sias_data[station_id].dropna()
        
        if len(series) < 365 * 2: # Need at least 2 years of data
            continue
            
        # Test period 1: up to 2009
        series_p1 = series[series.index.year <= SIAS_PERIOD_1_END]
        tau1, p1 = pettitt_test(series_p1.values)
        break_date1 = series_p1.index[tau1] if p1 < ALPHA else None

        # Test period 2: 2009 to 2013
        series_p2 = series[(series.index.year > SIAS_PERIOD_1_END) & (series.index.year <= SIAS_PERIOD_2_END)]
        if not series_p2.empty:
            tau2, p2 = pettitt_test(series_p2.values)
            break_date2 = series_p2.index[tau2] if p2 < ALPHA else None
        else:
            break_date2 = None
            
        results.append({
            'station_id': station_id,
            'break_found_period1 (->2009)': break_date1,
            'break_found_period2 (2009-2013)': break_date2,
        })
        
    return pd.DataFrame(results)