library(xcms)       # Load XCMS
library(RMassBank)  # Load RMassBank
library(gtools)     # Used for natural sorting
library(yaml)       # Used for reading configuration file

start_feature_extraction <- function(config_filename) {

    # load the yml configuration file
    config <- yaml.load_file(config_filename)
    
    ##########################
    ##### Peak Detection #####
    ##########################
    
    create_peak_method <- config$create_peak_method
    if (create_peak_method == 1) {    
        print("Running create_peak_method #1")        
        source('runCreatePeakMethod1.R')    
        peaks <- run_create_peak_method_1(config)
    } else if (create_peak_method == 2) {
        print("Running create_peak_method #2")
        source('runCreatePeakMethod2.R')    
        peaks <- run_create_peak_method_2(config)
    } else if (create_peak_method == 3) {
        print("Running create_peak_method #3")        
        source('runCreatePeakMethod3.R')    
        peaks <- run_create_peak_method_3(config)    
    }
    
    ###############################
    ##### Feature Extractions #####
    ###############################
    
    # do further filtering inside create_peaklist() method
    source('createPeakList.R')
    results <- create_peaklist(peaks, config)
    ms1 <- results$ms1
    ms2 <- results$ms2
    
    source('extractFragmentFeatures.R')
    results <- extract_ms2_fragment_df(ms1, ms2, config)
    fragment_df <- results$fragment_df
    ms2 <- results$ms2
    
    source('extractLossFeatures.R')
    results <- extract_neutral_loss_df(ms1, ms2, config)
    neutral_loss_df <- results$neutral_loss_df
    ms2 <- results$ms2
    loss_values_df <- results$loss_values_df
    
    # source('extractMzdiffFeatures.R')
    # results <- extract_mzdiff_df(ms1, ms2, config)
    # mz_diff_df <- results$mz_diff_df
    # ms2 <- results$ms2
    
    # post-processing: for losses<40, merge rows within 0.01 Dalton together
    source('postProcessing.R')
    results <- post_process_neutral_loss(neutral_loss_df, loss_values_df, ms2, config)   
    neutral_loss_df <- results$neutral_loss_df
    ms2 <- results$ms2
    
    ########################
    ##### Write Output #####
    ########################
    
    source('writeDataframes.R')
    write_output(ms1, ms2, fragment_df, neutral_loss_df, config)
        
}