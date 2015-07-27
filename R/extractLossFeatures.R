library(gtools) # for natural sorting

extract_neutral_loss_df <- function(ms1, ms2, prev_words_file) {
    
    #############################################
    ##### Neutral Loss Dataframe Generation #####
    #############################################
    
    print("Constructing neutral loss dataframe")

    # create an empty dataframe for existing words
    existing_loss_df <- data.frame(t(rep(NA,length(ms1$peakID))))
    existing_loss_df <- existing_loss_df[-1,] # remove first column

    # create an empty dataframe for new words
    new_loss_df <- data.frame(t(rep(NA,length(ms1$peakID))))
    new_loss_df <- new_loss_df[-1,] # remove first column
    
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
    
    # reuse existing words, if any
    if (file.exists(prev_words_file)) {

        # select the rows containing 'loss' and make it into a vector for strsplit
        prev_words <- read.csv(prev_words_file, header=F)
        pos <-with(prev_words, grepl("loss", V1))
        prev_words <- as.vector(prev_words[pos, 1]) 
        
        # split by _
        tokens <- strsplit(prev_words, '_')
        prev_mzs <- sapply(tokens, '[', 2)
        prev_mzs <- as.numeric(prev_mzs)
        prev_mzs_len <- length(prev_mzs)
        
        for (i in 1:prev_mzs_len) {
            
            mz <- prev_mzs[i]
            
            # calculate mz window
            max.ppm <- mz * 15 * 1e-06
            
            # find losses within that window
            match.idx <- which(sapply(losses, function(x) {
                abs(mz - x) < abs(max.ppm)
            }))    
            
            # use the existing word as label for the rows
            mean.mz <- round(mz, digits=5)
            
            # if there's a match then use the actual fragment peaks
            if (length(match.idx)>0) { 
                            
                # find column of the parent peaks
                intensities <- fragment_intensities[match.idx]
                peakids <- fragment_peakids[match.idx]
                parent.id <- parent_ids[match.idx]
                parent.idx <- match(as.character(parent.id), ms1.names)
                
                # append this new row to the data frame only if no. of parent.idx > threshold
                threshold_counts <- 5
                threshold_max_loss <- 200
                if (length(parent.idx) >= threshold_counts && mean.mz < threshold_max_loss) {
                    
                    print(paste(c("i=", i, "/", prev_mzs_len, 
                                  ", remaining=", length(losses), 
                                  " loss=", mean.mz, 
                                  " matches=", length(match.idx)), collapse=""))
                    row <- rep(NA, nrow(ms1))
                    row[parent.idx] <- intensities
                    
                    # add new row to existing loss dataframe
                    existing_loss_df <- rbind(existing_loss_df, row)
                    rownames(existing_loss_df)[nrow(existing_loss_df)] <- paste(c("loss_", mean.mz), collapse="") # the row name is the avg mz
                    
                    # store the loss_bin_id in the original ms2 dataframe too
                    matching_pos <- match(as.character(peakids), ms2.names)
                    ms2[matching_pos, "loss_bin_id"] <- as.character(mean.mz) 
                
                    # remove losses from the list to process
                    losses <- losses[-match.idx]
                    fragment_peakids <- fragment_peakids[-match.idx]
                    fragment_intensities <- fragment_intensities[-match.idx]
                    parent_ids <- parent_ids[-match.idx]
                    
                } else { # otherwise just insert a row of all NAs   

                    row <- rep(NA, nrow(ms1))
                    existing_loss_df <- rbind(existing_loss_df, row)
                    rownames(existing_loss_df)[nrow(existing_loss_df)] <- paste(c("loss_", mean.mz), collapse="")
                    
                }
                
            } else { # otherwise just insert a row of all NAs   
                
                row <- rep(NA, nrow(ms1))
                existing_loss_df <- rbind(existing_loss_df, row)
                rownames(existing_loss_df)[nrow(existing_loss_df)] <- paste(c("loss_", mean.mz), collapse="")
                
            }            
            
        }
        
    }
        
    # greedily discretise the remaining loss values
    # remember that we want to group similar losses values together too
    while(length(losses) > 0) {
                
        mz <- losses[1]
        
        # get all the losses values within tolerance from mz
        max.ppm <- mz * 15 * 1e-06
        match.idx <- which(sapply(losses, function(x) {
            abs(mz - x) < abs(max.ppm) # compare against abs(max.ppm) just in case there's MS2 mz > MS1 mz
        }))    
        
        stopifnot(length(match.idx) > 0) # we must always find something here ..
        
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

            print(paste(c("remaining=", length(losses), " loss=", mean.mz, " matches=", length(match.idx), " accepted"), collapse=""))
            
            row <- rep(NA, nrow(ms1))
            row[parent.idx] <- intensities
            
            new_loss_df <- rbind(new_loss_df, row)
            rownames(new_loss_df)[nrow(new_loss_df)] <- paste(c("loss_", mean.mz), collapse="") # the row name is the avg mz
            
            matching_pos <- match(as.character(peakids), ms2.names)
            ms2[matching_pos, "loss_bin_id"] <- as.character(mean.mz)            
            
        } else {
            print(paste(c("remaining=", length(losses), " loss=", mean.mz, " matches=", length(match.idx), " rejected"), collapse=""))
        }
        
        # this will always find something -- following the assertion in stopifnot() above
        losses <- losses[-match.idx]
        fragment_peakids <- fragment_peakids[-match.idx]
        fragment_intensities <- fragment_intensities[-match.idx]
        parent_ids <- parent_ids[-match.idx]
                
    }
    
    existing_loss_df <- existing_loss_df[mixedsort(row.names(existing_loss_df)),]
    names(existing_loss_df) <- paste(as.character(round(ms1$mz, digits=5)), 
                                         as.character(ms1$rt),
                                         as.character(ms1$peakID),
                                         sep="_")
    
    new_loss_df <- new_loss_df[mixedsort(row.names(new_loss_df)),]
    names(new_loss_df) <- paste(as.character(round(ms1$mz, digits=5)), 
                                    as.character(ms1$rt),
                                    as.character(ms1$peakID),
                                    sep="_")
    
    loss_df <- rbind(existing_loss_df, new_loss_df)    
    
    # ms2 has been modified and needs to be returned too
    output <- list("neutral_loss_df"=loss_df, "ms2"=ms2)
    return(output)
        
}
