library(xcms)
library(Hmisc)
library(gtools)

# beer3 and urine37 dataset
# input_file <- '/home/joewandy/Project/justin_data/Beer_3_T10_POS.mzXML'
# input_file <- '/home/joewandy/Project/justin_data/Beer_3_T10_NEG.mzXML'
# input_file <- '/home/joewandy/Project/justin_data/Urine_37_Top10_POS.mzXML'
# input_file <- '/home/joewandy/Project/justin_data/Urine_37_Top10_NEG.mzXML'

# beer2pos and beer3pos as training-testing data
input_file <- '/home/joewandy/Project/justin_data/Beer_data/Positive/Beer_2_T10_POS.mzXML'
# input_file <- '/home/joewandy/Project/justin_data/Beer_data/Positive/Beer_3_T10_POS.mzXML'

# reuse prev vocabularies, if any
# prev_words_file <- '/home/joewandy/git/metabolomics_tools/justin/input/test.selected.words'

# construct the output filenames
prefix <- basename(input_file) # get the filename only
prefix <- sub("^([^.]*).*", "\\1", prefix) # get rid of the extension 

# if true, we will use the relative intensities of the ms2 peaks instead of absolute intensity
use_relative_intensities <- TRUE
if (use_relative_intensities) {
    fragments_out <- paste(c(prefix, '_fragments_rel.csv'), collapse="")
    losses_out <- paste(c(prefix, '_losses_rel.csv'), collapse="")
    mzdiffs_out <- paste(c(prefix, '_mzdiffs_rel.csv'), collapse="")
    ms1_out <- paste(c(prefix, '_ms1_rel.csv'), collapse="")
    ms2_out <- paste(c(prefix, '_ms2_rel.csv'), collapse="")
} else {
    fragments_out <- paste(c(prefix, '_fragments.csv'), collapse="")
    losses_out <- paste(c(prefix, '_losses.csv'), collapse="")
    mzdiffs_out <- paste(c(prefix, '_mzdiffs.csv'), collapse="")
    ms1_out <- paste(c(prefix, '_ms1.csv'), collapse="")
    ms2_out <- paste(c(prefix, '_ms2.csv'), collapse="")   
}

################################
## Read in data and get peaks ##
################################

# do peak detection using CentWave
xset <- xcmsSet(files=input_file, method="centWave", ppm=2, snthresh=3, peakwidth=c(5,100),
                prefilter=c(3,1000), mzdiff=0.001, integrate=0, fitgauss=FALSE, verbose.column=TRUE)
xset <- group(xset)

# load Tony Larson's script
source('xcmsSetFragments.R')
frags <- xcmsSetFragments(xset, cdf.corrected = FALSE, min.rel.int=0.01, max.frags = 5000, 
                          msnSelect=c("precursor_int"), specFilter=c("specPeaks"), match.ppm=7, 
                          sn=3, mzgap=0.005, min.r=0.75, min.diff=10)
peaks <- as.data.frame(frags@peaks)

##########################
##### Data filtering #####
##########################

source('createPeakList.R')
results <- create_peaklist(peaks, use_relative_intensities)
ms1 <- results$ms1
ms2 <- results$ms2

###############################
##### Feature Extractions #####
###############################

source('extractFragmentFeatures.R')
results <- extract_ms2_fragment_df(ms1, ms2, prev_words_file)
fragment_df <- results$fragment_df
ms2 <- results$ms2

source('extractLossFeatures.R')
results <- extract_neutral_loss_df(ms1, ms2, prev_words_file)
neutral_loss_df <- results$neutral_loss_df
ms2 <- results$ms2

source('extractMzdiffFeatures.R')
results <- extract_mzdiff_df(ms1, ms2)
mz_diff_df <- results$mz_diff_df
ms2 <- results$ms2

########################
##### Write Output #####
########################

write.table(ms1, file=ms1_out, col.names=NA, row.names=T, sep=",")
write.table(ms2, file=ms2_out, col.names=NA, row.names=T, sep=",")    
write.table(fragment_df, file=fragments_out, col.names=NA, row.names=T, sep=",")
write.table(neutral_loss_df, file=losses_out, col.names=NA, row.names=T, sep=",")
write.table(mz_diff_df, file=mzdiffs_out, col.names=NA, row.names=T, sep=",")
