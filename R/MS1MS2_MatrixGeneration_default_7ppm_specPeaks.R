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
prev_fragment_file <- 'Beer_3_T10_POS_fragments_rel.csv'
prev_loss_file <- 'Beer_3_T10_POS_losses_rel.csv'
prev_mzdiff_file <- 'Beer_3_T10_POS_mzdiffs_rel.csv'

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

### MS1 ###

# get ms1 peaks
ms1 <- peaks[which(peaks$msLevel==1),]

# keep peaks with RT > 3 mins and < 21 mins
ms1 <- ms1[which(ms1$rt >= 3*60),]
ms1 <- ms1[which(ms1$rt <= 21*60),]

### MS2 ###

# get ms2 peaks
ms2 <- peaks[which(peaks$msLevel==2),]

# keep ms2 peaks with intensity > 5000
ms2 <- ms2[which(ms2$intensity>5000),]

# keep ms2 peaks with parent in filtered ms1 list
ms2 <- ms2[which(ms2$MSnParentPeakID %in% ms1$peakID),]

# make sure only ms1 peaks with ms2 fragments are kept
ms1 <- ms1[which(ms1$peakID %in% ms2$MSnParentPeakID),]

# scale the intensities of ms2 peaks to relative intensity
if (use_relative_intensities) {
    
    parent_ids <- ms2$MSnParentPeakID
    for (i in 1:nrow(ms1)) {
        peak_id <- ms1[i, 1]
        matches <- match(as.character(parent_ids), peak_id)
        pos <- which(!is.na(matches))
        # if there's more than one fragment peak
        if (length(pos)>0) {
            # then scale by the relative intensities of the spectrum
            fragment_peaks <- ms2[pos, ]
            fragment_intensities <- fragment_peaks$intensity
            max_intense <- max(fragment_intensities)
            fragment_intensities <- fragment_intensities / max_intense
            ms2[pos, ]$intensity <- fragment_intensities
        }
    }
    
}

### Prepare the matrices for LDA ###

###############################
##### Feature Extractions #####
###############################

source('extractFragmentFeatures.R')
results <- extract_ms2_fragment_df(ms1, ms2, prev_fragment_file)
fragment_df <- results$fragment_df
ms2 <- results$ms2

source('extractLossFeatures.R')
results <- extract_neutral_loss_df(ms1, ms2)
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