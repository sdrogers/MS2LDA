require('gtools') # for natural sorting

extract_neutral_loss_df <- function(ms1, ms2) {
    
    #############################################
    ##### Neutral Loss Dataframe Generation #####
    #############################################
    
    print("Constructing neutral loss dataframe")
    
    # create empty data.frame
    neutral_loss_df <- data.frame(t(rep(NA,length(ms1$peakID))))
    neutral_loss_df <- neutral_loss_df[-1,]
    
    ms1.names <- as.character(ms1$peakID) # set row names on ms1 dataframe
    ms2.names <- as.character(ms2$peakID) # set row names on ms2 dataframe    
    
    # compute the difference between each fragment peak to its parent
    ms2_masses <- ms2$mz
    parent_ids <- ms2$MSnParentPeakID
    matches <- match(as.character(parent_ids), ms1.names)
    parent_masses <- ms1[matches, 5] # column 5 is the mz
    losses <- parent_masses - ms2_masses
    fragment_intensities <- ms2$intensity
    fragment_peakids <- ms2$peakID
    
    # greedily discretise the loss values
    while(length(losses) > 0) {
        
        mz <- losses[1]
        
        # get all the losses values within tolerance from mz
        max.ppm <- mz * 15 * 1e-06
        match.idx <- which(sapply(losses, function(x) {
            abs(mz - x) < max.ppm
        }))    
        
        # compute their average mean mz as the row label and find column of the parent peaks
        mean.mz <- round(mean(losses[match.idx]), digits=5)
        intensities <- fragment_intensities[match.idx]
        peakids <- fragment_peakids[match.idx]
        parent.id <- parent_ids[match.idx]
        parent.idx <- match(as.character(parent.id), ms1.names)
        
        # append this new row to the data frame only if no. of parent.idx > threshold
        threshold_counts <- 5
        threshold_max_loss <- 200
        if (length(parent.idx) >= threshold_counts && mean.mz < threshold_max_loss) {
            
            print(paste(c("remaining=", length(losses), " loss=", mean.mz, " matches=", length(match.idx)), collapse=""))
            row <- rep(NA, nrow(ms1))
            row[parent.idx] <- intensities
            
            neutral_loss_df <- rbind(neutral_loss_df, row)
            rownames(neutral_loss_df)[nrow(neutral_loss_df)] <- paste(c("loss_", mean.mz), collapse="") # the row name is the avg mz
            
            matching_pos <- match(as.character(peakids), ms2.names)
            ms2[matching_pos, "loss_bin_id"] <- as.character(mean.mz)            
            
        }
        
        # decrease items from the vectors
        losses <- losses[-match.idx]
        fragment_peakids <- fragment_peakids[-match.idx]
        fragment_intensities <- fragment_intensities[-match.idx]
        parent_ids <- parent_ids[-match.idx]
        
    }
    
    # add ms1 label in format mz_rt_peakid
    names(neutral_loss_df) <- paste(as.character(round(ms1$mz, digits=5)), 
                                    as.character(ms1$rt),
                                    as.character(ms1$peakID),
                                    sep="_")    
    
    neutral_loss_df <- neutral_loss_df[mixedsort(row.names(neutral_loss_df)),]

    # ms2 has been modified and needs to be returned too
    output <- list("neutral_loss_df"=neutral_loss_df, "ms2"=ms2)
    return(output)
        
}
