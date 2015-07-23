# Set the method you want to use for peak detection:
# - 1 is using Tony's script
# - 2 is using the modified Tony's script where we can specify the full scan data for MS1
# - 3 is using RMassBank's script
create_peak_method <- 3

# Check inside each sourced script for all the parameters etc.
if (create_peak_method == 1) {
    source('runCreatePeakMethod1.R')    
} else if (create_peak_method == 2) {
    source('runCreatePeakMethod2.R')    
} else if (create_peak_method == 3) {
    source('runCreatePeakMethod3.R')    
}

###############################
##### Feature Extractions #####
###############################

# do further filtering inside create_peaklist() method
source('createPeakList.R')
results <- create_peaklist(peaks)
ms1 <- results$ms1
ms2 <- results$ms2

# reuse prev vocabularies, if any .. for LDA.
# prev_words_file <- '/home/joewandy/git/metabolomics_tools/justin/notebooks/results/beer3_pos_rel/beer3pos.vocab'
prev_words_file <- ''

source('extractFragmentFeatures.R')
results <- extract_ms2_fragment_df(ms1, ms2, prev_words_file)
fragment_df <- results$fragment_df
ms2 <- results$ms2

source('extractLossFeatures.R')
results <- extract_neutral_loss_df(ms1, ms2, prev_words_file)
neutral_loss_df <- results$neutral_loss_df
ms2 <- results$ms2

# source('extractMzdiffFeatures.R')
# results <- extract_mzdiff_df(ms1, ms2)
# mz_diff_df <- results$mz_diff_df
# ms2 <- results$ms2

########################
##### Write Output #####
########################

# construct the output filenames
prefix <- basename(input_file) # get the filename only
prefix <- sub("^([^.]*).*", "\\1", prefix) # get rid of the extension 
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