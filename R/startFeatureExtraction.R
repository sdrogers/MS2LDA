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
    
    # construct the output filenames
    prefix <- config$input_files$prefix
    fragments_out <- paste(c(prefix, '_fragments.csv'), collapse="")
    losses_out <- paste(c(prefix, '_losses.csv'), collapse="")
    # mzdiffs_out <- paste(c(prefix, '_mzdiffs.csv'), collapse="")
    ms1_out <- paste(c(prefix, '_ms1.csv'), collapse="")
    ms2_out <- paste(c(prefix, '_ms2.csv'), collapse="")
    
    # write stuff out
    write.table(ms1, file=ms1_out, col.names=NA, row.names=T, sep=",")
    write.table(ms2, file=ms2_out, col.names=NA, row.names=T, sep=",")    
    write.table(fragment_df, file=fragments_out, col.names=NA, row.names=T, sep=",")
    write.table(neutral_loss_df, file=losses_out, col.names=NA, row.names=T, sep=",")
    # write.table(mz_diff_df, file=mzdiffs_out, col.names=NA, row.names=T, sep=",")
    
}