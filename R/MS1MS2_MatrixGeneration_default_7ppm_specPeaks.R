library(xcms)
library(Hmisc)
library(gtools)

# input_file <- '/home/joewandy/Project/justin_data/Beer_3_T10_POS.mzXML'
# input_file <- '/home/joewandy/Project/justin_data/Beer_3_T10_NEG.mzXML'
# input_file <- '/home/joewandy/Project/justin_data/Urine_37_Top10_POS.mzXML'
input_file <- '/home/joewandy/Project/justin_data/Urine_37_Top10_NEG.mzXML'

# construct the output filenames
prefix <- basename(input_file) # get the filename only
prefix <- sub("^([^.]*).*", "\\1", prefix) # get rid of the extension 
fragments_out <- paste(c(prefix, '_fragments.csv'), collapse="")
losses_out <- paste(c(prefix, '_losses.csv'), collapse="")
mzdiffs_out <- paste(c(prefix, '_mzdiffs.csv'), collapse="")
ms1_out <- paste(c(prefix, '_ms1.csv'), collapse="")
ms2_out <- paste(c(prefix, '_ms2.csv'), collapse="")

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

### Prepare the matrices for LDA ###

source('extractFeatures.R')
extract_features(ms1, ms2, ms1_out, ms2_out, 
                 fragments_out, losses_out, mzdiffs_out)
