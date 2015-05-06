require('gtools') # for natural sorting

# TODO: tidy up and speed up this function later ...
extract_features <- function(ms1, ms2, ms1_out, ms2_out, 
                             fragments_out, losses_out, mzdiffs_out) {

    ########################################
    ##### MS1/MS2 Dataframe Generation #####
    ########################################
    
    print("Constructing MS1/MS2 dataframe")
    
    # create empty data.frame
    ms2_df <- data.frame(t(rep(NA,length(ms1$peakID))))
    
    ms1.names <- as.character(ms1$peakID) # set row names on ms1 dataframe
    ms2.names <- as.character(ms2$peakID) # set row names on ms2 dataframe
    ms2_df <- ms2_df[-1,] # remove first column
    
    # find fragments that are within 7ppm of each other. Assume same fragment.
    copy_ms2 <- ms2
    while(nrow(copy_ms2) > 0) {
        
        print(paste(c("remaining=", nrow(copy_ms2)), collapse=""))
        
        # get first mz value
        mz <- copy_ms2$mz[1]
        
        # calculate mz window
        max.ppm <- mz * 7 * 1e-06
        
        # find peaks within that window
        match.idx <- which(sapply(copy_ms2$mz, function(x) {
            abs(mz - x) < max.ppm
        }))    
        
        # calculate mean mz as label for ms2 row
        mean.mz <- round(mean(copy_ms2$mz[match.idx]), digits=5)
        
        # store the mean mz (bin id) into the original ms2 dataframe too
        peakids <- copy_ms2$peakID[match.idx]
        matching_pos <- match(as.character(peakids), ms2.names)
        ms2[matching_pos, "fragment_bin_id"] <- as.character(mean.mz)
        
        # get intensities
        intensities <- copy_ms2$intensity[match.idx]
        
        # get parent id
        parent.id <- copy_ms2$MSnParentPeakID[match.idx]
        
        # find parent id in data.frame and add ms2 fragments
        parent.idx <- match(as.character(parent.id), ms1.names)
        row <- rep(NA, nrow(ms1))
        row[parent.idx] <- intensities
        ms2_df <- rbind(ms2_df, row)
        rownames(ms2_df)[nrow(ms2_df)] <- paste(c("fragment_", mean.mz), collapse="")
        
        # remove fragments from ms2 list and start loop again with next fragment
        copy_ms2 <- copy_ms2[-match.idx,]
        
    }
    
    # add ms1 label in format mz_rt_peakid
    names(ms2_df) <- paste(as.character(round(ms1$mz, digits=5)), 
                           as.character(ms1$rt),
                           as.character(ms1$peakID),
                           sep="_")
    
    # sort in a natural order
    ms2_df <- ms2_df[mixedsort(row.names(ms2_df)),]
    
    #############################################
    ##### Neutral Loss Dataframe Generation #####
    #############################################
    
    print("Constructing neutral loss dataframe")
    
    # create empty data.frame
    neutral_loss_df <- data.frame(t(rep(NA,length(ms1$peakID))))
    neutral_loss_df <- neutral_loss_df[-1,]
    
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
        threshold <- 5
        if (length(parent.idx) >= threshold) {
            
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
    
    ########################
    ##### Write Output #####
    ########################
    
    write.table(ms1, file=ms1_out, col.names=NA, row.names=T, sep=",")
    write.table(ms2, file=ms2_out, col.names=NA, row.names=T, sep=",")    
    write.table(ms2_df, file=fragments_out, col.names=NA, row.names=T, sep=",")
    write.table(neutral_loss_df, file=losses_out, col.names=NA, row.names=T, sep=",")
    write.table(mz_diff_df, file=mzdiffs_out, col.names=NA, row.names=T, sep=",")
    
}
