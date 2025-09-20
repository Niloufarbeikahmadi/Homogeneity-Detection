# -*- coding: utf-8 -*-
"""
Configuration file for the rainfall data homogeneity project.

@author: Niloufar Beikahmadi
"""

# --- File Paths ---
# ODA Datasets
ODA_STATIONS_PATH = 'StatOdAxNil.csv'
ODA_DATA_PATH = 'OdAxNil.csv'

# SIAS Datasets
SIAS_STATIONS_PATH = 'stazioni_SIAS_def_33n.csv'
SIAS_DATA_PATH = 'cubbed_SIAS.nc' 

# Geospatial Data
SICILY_SHAPEFILE_PATH = 'sicily_shapefile.shp' #  the path to your shapefile

# --- Data Parameters ---
START_YEAR_ODA = 1950
END_YEAR = 2022
START_YEAR_SIAS = 2002

# --- Analysis Parameters ---
# Number of nearest neighbors to consider for each station
NUM_NEIGHBORS = 5 
# Significance level for the homogeneity tests
ALPHA = 0.05
# Window size for the SNHT test (in years)
SNHT_WINDOW = 10

# --- SIAS Auto-homogeneity Test Periods ---
SIAS_PERIOD_1_END = 2009
SIAS_PERIOD_2_END = 2013
