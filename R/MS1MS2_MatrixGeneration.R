# All the parameters to the workflow are stored in a YAML format, so modify config.yml as needed. 
# The default settings provided are used for stepped collision energy MS2 spectra of a 
# Thermo-Exactive, following pHilic chromatography.
# Ensure correct input files (including path if applicable) is filled out for the appropriate 
# peak picking method (1, 2, or 3).
config_filename <- "config.yml"

# Calls the primary script for the feature extraction workflow.
# This will perform peak detection on the mzXML fragmentation and/or full scan data (depending
# on the configuration parameters, perform the linking between the MS1 to MS2 peaks, groups
# similar masses together to create fragment/loss 'words' and populate the matrices of word
# counts, which can then be passed to LDA.

source('startFeatureExtraction.R')
start_feature_extraction(config_filename)