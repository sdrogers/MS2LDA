### This is the initial peak detection workflow we have froom before --- using Tony's script ###

library(xcms)
library(Hmisc)

run_create_peak_method_1 <- function(fragmentation_file) {

    print("Running create_peak_method #1")
    
    # do peak detection using CentWave
    xset <- xcmsSet(files=fragmentation_file, method="centWave", ppm=2, snthresh=3, peakwidth=c(5,100),
                    prefilter=c(3,1000), mzdiff=0.001, integrate=0, fitgauss=FALSE, verbose.column=TRUE)
    xset <- group(xset)
    
    # run Tony's script
    source('xcmsSetFragments.R')
    frags <- xcmsSetFragments(xset, cdf.corrected = FALSE, min.rel.int=0.01, max.frags = 5000, 
                              msnSelect=c("precursor_int"), specFilter=c("specPeaks"), match.ppm=7, 
                              sn=3, mzgap=0.005, min.r=0.75, min.diff=10)
    peaks <- as.data.frame(frags@peaks)
    
    return(peaks)
    
}