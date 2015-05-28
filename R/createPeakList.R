create_peaklist <- function(peaks, use_relative_intensities) {
    
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
    
    # scale the intensities of ms2 peaks to relative intensity?
    if (use_relative_intensities) {
        
        parent_ids <- ms2$MSnParentPeakID
        for (i in 1:nrow(ms1)) {
            
            print(paste(c("i=", i, "/", nrow(ms1)), collapse=""))
            
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
    
    output <- list("ms1"=ms1, "ms2"=ms2)
    return(output)
    
}
