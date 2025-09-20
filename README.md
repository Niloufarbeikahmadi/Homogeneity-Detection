Rainfall Data Homogeneity Analysis
This project provides a Python framework for testing the homogeneity of rainfall time series data from two different networks (ODA and SIAS) and classifying them based on the results.
Project Structure
●	main.py: The main script to run the entire analysis.
●	data_loader.py: Handles loading and preprocessing of ODA and SIAS data.
●	homogeneity_tests.py: Contains implementations of the Pettitt test and SNHT.
●	analysis.py: Includes functions for finding neighbors, running the tests on pairwise difference series, and classifying stations.
●	config.py: A centralized configuration file for paths and parameters.
●	requirements.txt: A list of required Python packages.
●	StatOdAxNil.csv, OdAxNil.csv, stazioni_SIAS_def_33n.csv: Your data files.
●	sicily_shapefile.shp (and related files like .dbf, .shx): Your shapefile for filtering stations.
Setup
1.	Place your data files in the same directory as the scripts:
○	StatOdAxNil.csv
○	OdAxNil.csv
○	stazioni_SIAS_def_33n.csv
○	Your Sicily shapefile (e.g., sicily_shapefile.shp, sicily_shapefile.dbf, etc.)
2.	Install the required libraries:
pip install -r requirements.txt

3.	Configure your settings in config.py. You can adjust parameters like the number of neighbors, significance level, and file paths if they are different.
4.	IMPORTANT - SIAS Data Loading: The data_loader.py script currently simulates the SIAS rainfall data. You must replace the simulation block with your actual code to load and process your cubbed_SIAS.nc file.
How to Run
Execute the main script from your terminal:
python main.py

Output
The script will produce two CSV files:
1.	homogeneity_classification.csv: This file contains the results of the relative homogeneity tests. Each station is classified as 'Useful', 'Doubtful', or 'Suspect'.
○	station_id: The ID of the station.
○	pettitt_rejects: Number of neighbors for which the Pettitt test rejected the null hypothesis.
○	snht_rejects: Number of neighbors for which the SNHT rejected the null hypothesis.
○	total_rejects: The sum of rejections.
○	classification: The final classification of the station.
2.	sias_auto_homogeneity_results.csv: This file shows the results of the internal homogeneity check for the SIAS stations, specifically looking for breaks around the times the instrumentation was changed.
○	station_id: The ID of the SIAS station.
○	break_found_period1 (->2009): The date of a detected break point before the end of 2009. NaT if no significant break was found.
○	break_found_period2 (2009-2013): The date of a detected break point between 2009 and 2013. NaT if no significant break was found.
This comprehensive set of scripts will allow you to systematically assess the quality and consistency of your rainfall data before merging the two datasets.
