library(gtools) # for natural sorting

extract_neutral_loss_df <- function(ms1, ms2) {
    
    prev_words_file <- config$input_files$previous_words_file
    grouping_tol <- config$MS1MS2_matrixGeneration_parameters$grouping_tol_losses
    threshold_counts <- config$MS1MS2_matrixGeneration_parameters$threshold_counts
    threshold_max_loss <- config$MS1MS2_matrixGeneration_parameters$threshold_max_loss
  
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

    # store the actual grouped values of the losses too
    existing_loss_values <- vector()
    new_loss_values <- vector()
    
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

    total_rejected <- 0
    total_accepted <- 0
    
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

            # should use abs() here just in case there's MS2 mz > MS1 mz
            max_ppm <- abs(mz*grouping_tol*1e-06)
            # match to the first unmatched peak
            temp <- abs(mz-losses)
            match_idx <- which(temp <= abs(max_ppm))
            
            # use the existing word as label for the rows
            mean_mz <- round(mz, digits=5)
            
            # if there's a match then use the actual fragment peaks
            if (length(match_idx)>0) { 
                            
                # find column of the parent peaks
                intensities <- fragment_intensities[match_idx]
                peakids <- fragment_peakids[match_idx]
                parent_id <- parent_ids[match_idx]
                parent_idx <- match(as.character(parent_id), ms1.names)
                
                # append this new row to the data frame only if no. of parent_idx > threshold
                if (length(parent_idx) >= threshold_counts && mean_mz < threshold_max_loss) {
                    
                    print(paste(c("i=", i, "/", prev_mzs_len, 
                                  ", remaining=", length(losses), 
                                  " loss=", mean_mz, 
                                  " matches=", length(match_idx)), " accepted", collapse=""))
                    total_accepted <- total_accepted + 1
                    
                    row <- rep(NA, nrow(ms1))
                    row[parent_idx] <- intensities
                    
                    # each row must have at least one non-na value
                    non_na_pos <- length(which(!is.na(row)))
                    stopifnot(length(non_na_pos) == 0)                    
                    
                    # add new row to existing loss dataframe
                    existing_loss_df <- rbind(existing_loss_df, row)

                    # the row name is the avg mz
                    rownames(existing_loss_df)[nrow(existing_loss_df)] <- paste(c("loss_", mean_mz), 
                                                                                collapse="")
                    
                    # store the loss_bin_id in the original ms2 dataframe too
                    matching_pos <- match(as.character(peakids), ms2.names)
                    ms2[matching_pos, "loss_bin_id"] <- as.character(mean_mz)   
                    existing_loss_values <- c(existing_loss_values, mean_mz)    
                    
                    # remove losses from the list to process
                    losses <- losses[-match_idx]
                    fragment_peakids <- fragment_peakids[-match_idx]
                    fragment_intensities <- fragment_intensities[-match_idx]
                    parent_ids <- parent_ids[-match_idx]
                                                            
                } else { # otherwise just insert a row of all NAs   

                    print(paste(c("i=", i, "/", prev_mzs_len, 
                                  ", remaining=", length(losses), 
                                  " loss=", mean_mz, 
                                  " matches=", length(match_idx)), " rejected", collapse=""))
                    total_rejected <- total_rejected + 1
                    
                    row <- rep(NA, nrow(ms1))
                    existing_loss_df <- rbind(existing_loss_df, row)
                    rownames(existing_loss_df)[nrow(existing_loss_df)] <- paste(c("loss_", mean_mz), 
                                                                                collapse="")
                    
                }
                
            } else { # otherwise just insert a row of all NAs   

                print(paste(c("i=", i, "/", prev_mzs_len, 
                              ", remaining=", length(losses), 
                              " loss=", mean_mz, 
                              " matches=", length(match_idx)), " rejected", collapse=""))
                total_rejected <- total_rejected + 1
                
                row <- rep(NA, nrow(ms1))
                existing_loss_df <- rbind(existing_loss_df, row)
                rownames(existing_loss_df)[nrow(existing_loss_df)] <- paste(c("loss_", mean_mz), 
                                                                            collapse="")
                
            }            
            
        }
        
    }
        
    # greedily discretise the remaining loss values
    while(length(losses) > 0) {
                
        mz <- losses[1]
        
        # match to the initial peak
#         max_ppm <- mz * grouping_tol * 1e-06
#         match_idx <- which(sapply(losses, function(x) {
#             # compare against abs(max_ppm) just in case there's MS2 mz > MS1 mz
#             abs(mz - x) < abs(max_ppm)
#         }))

        # should use abs() here just in case there's MS2 mz > MS1 mz
        max_ppm <- abs(mz*grouping_tol*1e-06)
        # match to the first unmatched peak
        temp <- abs(mz-losses)
        match_idx <- which(temp <= abs(max_ppm))
        
#         # experimental code to try a more sophisticated grouping of the loss values
#         # doesn't seem to improve things that much ..
#
#         # set the current peak as the candidate
#         match_idx <- vector()
#         match_idx <- c(match_idx, 1)
#         # then try to match other peaks to the average of the candidates each time
#         if (length(losses) > 1) {
#             candidates <- vector()
#             candidates <- c(candidates, mz)
#             temp_mean <- mz
#             max_ppm <- temp_mean * grouping_tol * 1e-06
#             # find other candidates
#             for (i in 2:length(losses)) {
#                 x <- losses[i]
#                 # print(paste(c("x=", x, " temp_mean=", temp_mean, " max_ppm=", max_ppm), collapse=""))            
#                 if (abs(temp_mean-x) < abs(max_ppm)) {
#                     candidates <- c(candidates, x)
#                     match_idx <- c(match_idx, i)
#                     temp_mean <- mean(candidates) # and keep updating the centroid
#                     max_ppm <- temp_mean * grouping_tol * 1e-06                
#                 }
#             }
#         }
                        
        stopifnot(length(match_idx) > 0) # we must always find something here ..
        
        # compute their average mean mz as the row label and find column of the parent peaks
        mean_mz <- round(mean(losses[match_idx]), digits=5)
        intensities <- fragment_intensities[match_idx]
        peakids <- fragment_peakids[match_idx]
        parent_id <- parent_ids[match_idx]
        parent_idx <- match(as.character(parent_id), ms1.names)

        # append this new row to the data frame only if no. of parent_idx > threshold
        if (length(parent_idx) >= threshold_counts && mean_mz > 0 && mean_mz < threshold_max_loss) {
            
            print(paste(c("remaining=", length(losses), 
                          " loss=", mean_mz, " matches=", length(match_idx), " accepted"), 
                        collapse=""))
            total_accepted <- total_accepted + 1
            
            row <- rep(NA, nrow(ms1))
            row[parent_idx] <- intensities

            # each row must have at least one non-na value
            count_non_na <- length(which(!is.na(row)))
            stopifnot(count_non_na > 0) 
            
            # add row to dataframe
            new_loss_df <- rbind(new_loss_df, row)

            # the row name is the avg mz
            rownames(new_loss_df)[nrow(new_loss_df)] <- paste(c("loss_", mean_mz), 
                                                              collapse="") 
            
            matching_pos <- match(as.character(peakids), ms2.names)
            ms2[matching_pos, "loss_bin_id"] <- as.character(mean_mz)            
            new_loss_values <- c(new_loss_values, mean_mz)
            
        } else {
            print(paste(c("remaining=", length(losses), 
                          " loss=", mean_mz, " matches=", length(match_idx), " rejected"), 
                        collapse=""))
            total_rejected <- total_rejected + 1
        }
        
        # remove items that have been processed
        losses <- losses[-match_idx]
        fragment_peakids <- fragment_peakids[-match_idx]
        fragment_intensities <- fragment_intensities[-match_idx]
        parent_ids <- parent_ids[-match_idx]
                
    }
    
    print(paste(c("total_accepted=", total_accepted), collapse=""))
    print(paste(c("total_rejected=", total_rejected), collapse=""))
    
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
    
    neutral_loss_df <- rbind(existing_loss_df, new_loss_df)   
    
    # this is useful for post-processing later
    loss_values <- c(sort(existing_loss_values), sort(new_loss_values))
    loss_ids <- 1:length(loss_values)
    loss_values_df <- data.frame(id=loss_ids, loss=loss_values)

    # ms2 has been modified and needs to be returned too
    output <- list("neutral_loss_df"=neutral_loss_df, "ms2"=ms2, "loss_values_df"=loss_values_df)
    return(output)
        
}