### This is the initial peak detection workflow we have before --- using Tony's script ###
### but modified to allow us to specify which MS1 full scan data to use ###

library(xcms)
library(Hmisc)

print("Running create_peak_method #2")

## do peak detection on full scan file
full_scan_input_file <- '/home/joewandy/Dropbox/Project/justin_data/Dataset_for_PiMP/Beers_4Beers_compared/Positive/Samples/Beer_3_full1.mzXML'
xset_full <- xcmsSet(files=full_scan_input_file, method="centWave", ppm=2, snthresh=3, peakwidth=c(5,100),
                     prefilter=c(3,1000), mzdiff=0.001, integrate=0, fitgauss=FALSE, verbose.column=TRUE)
xset_full <- group(xset_full)

# do peak detection on fragmentation file
input_file <- '/home/joewandy/Project/justin_data/Beer_3_T10_POS.mzXML'
xset <- xcmsSet(files=input_file, method="centWave", ppm=2, snthresh=3, peakwidth=c(5,100),
                prefilter=c(3,1000), mzdiff=0.001, integrate=0, fitgauss=FALSE, verbose.column=TRUE)
xset <- group(xset)

# run modified Tony's script
source('xcmsSetFragments.modified.R')
frags <- xcmsSetFragments(xset, xset_full, cdf.corrected = FALSE, min.rel.int=0.01, max.frags = 5000, 
                          msnSelect=c("precursor_int"), specFilter=c("specPeaks"), match.ppm=7, 
                          sn=3, mzgap=0.005, min.r=0.75, min.diff=10)
peaks <- as.data.frame(frags@peaks)