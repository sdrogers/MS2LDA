require('gtools') # for natural sorting

extract_mzdiff_df <- function(ms1, ms2) {
    
    ##############################################
    ##### Mz Difference Dataframe Generation #####
    ##############################################
    
    print("Constructing mz difference dataframe")
    
    # for every parent peak, get the pairwise absolute difference between its fragments' mz values
    all_mzdiffs = vector()
    all_parentids = vector()
    parent_count <- nrow(ms1)
    parent_ids <- ms2$MSnParentPeakID
    for (i in 1:nrow(ms1)) {
        peak_id <- ms1[i, 1]
        matches <- match(as.character(parent_ids), peak_id)
        pos <- which(!is.na(matches))
        # if there's more than one fragment peak
        if (length(pos)>1) {
            # then enumerate all combinations of size 2 between the fragment peaks
            fragment_peaks <- ms2[pos, ]
            fragment_mzs <- fragment_peaks$mz
            pairwise_mat <- combn(fragment_mzs, 2) 
            # assign the column names to the pairwise_mat matrix, so we can sapply over the columns
            colnames(pairwise_mat) <- seq(1:ncol(pairwise_mat))
            # find the absolute difference between each pairs
            pairwise_diff <- sapply(colnames(pairwise_mat), function(x){ 
                abs(pairwise_mat[1, x] - pairwise_mat[2, x]) 
            })
            # store the results into mzdiffs for use later
            print(paste(c("#", i, "/", parent_count, " parent peak id ", peak_id, " has ", 
                          length(pos), " fragment peaks and ", length(pairwise_diff), 
                          " pairwise combinations"), collapse=""))
            all_mzdiffs <- c(all_mzdiffs, unname(pairwise_diff))
            all_parentids <- c(all_parentids, rep(peak_id, length(pairwise_diff)))
        }
    }
    
    # sort first
    idx <- order(all_mzdiffs)
    all_mzdiffs <- all_mzdiffs[idx]
    all_parentids <- all_parentids[idx]
    
    # keep only the mzdiffs above 15.9, because 16 is water
    idx <- which(all_mzdiffs>15.9)
    all_mzdiffs <- all_mzdiffs[idx]
    all_parentids <- all_parentids[idx]
    print(paste(c("total pairwise differences=", length(all_mzdiffs)), collapse=""))
    
    # greedily discretise the mz diff values and put into the dataframe
    mz_diff_df <- data.frame(t(rep(NA,length(ms1$peakID))))
    mz_diff_df <- mz_diff_df[-1,]
    ms1.names <- as.character(ms1$peakID) # set row names on ms1 dataframe
    ms2.names <- as.character(ms2$peakID) # set row names on ms2 dataframe    
    temp_count <- vector()
    while(length(all_mzdiffs) > 0) {
        
        mz <- all_mzdiffs[1]
        
        # get all the losses values within tolerance from mz
        max.ppm <- mz * 15 * 1e-06
        match.idx <- which(sapply(all_mzdiffs, function(x) {
            abs(mz - x) < max.ppm
        }))    
        
        matched_mzdiffs <- all_mzdiffs[match.idx]
        matched_parentids <- all_parentids[match.idx]
        mean.mz <- round(mean(matched_mzdiffs), digits=5)
        
        # for each parent peak, compute how often the differences occur
        temp_df <- as.data.frame(table(matched_parentids))
        grouped_parentids <- temp_df[, 1]
        grouped_freqs <- temp_df[, 2]
        
        threshold <- 5
        if (length(grouped_parentids)>=threshold) {
            
            # for histogram
            temp_count <- c(temp_count, length(grouped_parentids))
            print(paste(c("remaining=", length(all_mzdiffs), " mzdiff=", mean.mz, " matches=", 
                          length(grouped_parentids)), collapse=""))
            
            # find column of the parent peaks and add the new row to the data frame
            parent.idx <- match(as.character(grouped_parentids), ms1.names)
            row <- rep(NA, nrow(ms1))
            row[parent.idx] <- grouped_freqs
            mz_diff_df <- rbind(mz_diff_df, row)
            
            # the name of the new row is the avg mz of the differences
            rownames(mz_diff_df)[nrow(mz_diff_df)] <- paste(c("mzdiff_", mean.mz), collapse="")
            
        }                
        
        # decrease items from the vectors
        all_mzdiffs <- all_mzdiffs[-match.idx]
        all_parentids <- all_parentids[-match.idx]    
        
    }
    
    # add ms1 label in format mz_rt
    names(mz_diff_df) <- paste(as.character(round(ms1$mz, digits=5)), 
                               as.character(ms1$rt),
                               as.character(ms1$peakID),
                               sep="_")
    mz_diff_df <- mz_diff_df[mixedsort(row.names(mz_diff_df)),]
    print(paste(c("no. of rows in mz_diff_df=", nrow(mz_diff_df)), collapse=""))
    hist(temp_count, breaks=seq(5, max(temp_count), 1))
    
    # ms2 has been modified and needs to be returned too
    output <- list("mz_diff_df"=mz_diff_df, "ms2"=ms2)
    return(output)
    
}
