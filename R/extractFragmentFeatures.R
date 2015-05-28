require('gtools') # for natural sorting

extract_ms2_fragment_df <- function(ms1, ms2, prev_words_file) {
    
    ########################################
    ##### MS1/MS2 Dataframe Generation #####
    ########################################
    
    print("Constructing MS1/MS2 dataframe")
    
    # create an empty dataframe for existing words
    existing_fragment_df <- data.frame(t(rep(NA,length(ms1$peakID))))
    existing_fragment_df <- existing_fragment_df[-1,] # remove first column

    # create an empty dataframe for new words
    new_fragment_df <- data.frame(t(rep(NA,length(ms1$peakID))))
    new_fragment_df <- new_fragment_df[-1,] # remove first column
    
    ms1.names <- as.character(ms1$peakID) # set row names on ms1 dataframe
    ms2.names <- as.character(ms2$peakID) # set row names on ms2 dataframe
    
    # reuse existing words, if any
    copy_ms2 <- ms2
    if (file.exists(prev_words_file)) {

        prev_words <- read.csv(prev_words_file, header=F)
        pos <-with(prev_words, grepl("fragment", V1))
        
        # select the rows containing 'fragment' and make it into a vector for strsplit
        prev_words <- as.vector(prev_words[pos, 1]) 
        
        # split by _
        tokens <- strsplit(prev_words, '_')
        prev_mzs <- sapply(tokens, '[', 2)
        prev_mzs <- as.numeric(prev_mzs)
        prev_mzs_len <- length(prev_mzs)
        
        for (i in 1:prev_mzs_len) {

            print(paste(c("i=", i, "/", prev_mzs_len, 
                          ", remaining=", nrow(copy_ms2)), collapse=""))
            mz <- prev_mzs[i]
                        
            # calculate mz window
            max.ppm <- mz * 7 * 1e-06
            
            # find peaks within that window
            match.idx <- which(sapply(copy_ms2$mz, function(x) {
                abs(mz - x) < max.ppm
            }))    
            
            # use the existing word as label for the rows
            mean.mz <- round(mz, digits=5)

            # if there's a match then use the actual fragment peaks
            if (length(match.idx)>0) { 
            
                # store the mean mz (bin id) into the original ms2 dataframe too
                peakids <- copy_ms2$peakID[match.idx]
                matching_pos <- match(as.character(peakids), ms2.names)
                ms2[matching_pos, "fragment_bin_id"] <- as.character(mean.mz)
                
                # get intensities
                intensities <- copy_ms2$intensity[match.idx]
                
                # get parent id
                parent.id <- copy_ms2$MSnParentPeakID[match.idx]
                
                # add a row of the intensities of the fragments
                parent.idx <- match(as.character(parent.id), ms1.names)
                row <- rep(NA, nrow(ms1))
                row[parent.idx] <- intensities                
                existing_fragment_df <- rbind(existing_fragment_df, row)
                rownames(existing_fragment_df)[nrow(existing_fragment_df)] <- paste(c("fragment_", mean.mz), collapse="")
                
                # remove fragments from ms2 list and start loop again with next fragment
                copy_ms2 <- copy_ms2[-match.idx,]
                            
            } else { # otherwise just insert a row of all NAs   
            
                row <- rep(NA, nrow(ms1))
                existing_fragment_df <- rbind(existing_fragment_df, row)
                rownames(existing_fragment_df)[nrow(existing_fragment_df)] <- paste(c("fragment_", mean.mz), collapse="")
            
            }            
                                    
        }

    }
    
    # then process the remaining fragments that have not been discretised yet
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
        new_fragment_df <- rbind(new_fragment_df, row)
        rownames(new_fragment_df)[nrow(new_fragment_df)] <- paste(c("fragment_", mean.mz), collapse="")
        
        # remove fragments from ms2 list and start loop again with next fragment
        copy_ms2 <- copy_ms2[-match.idx,]
        
    }
    
    existing_fragment_df <- existing_fragment_df[mixedsort(row.names(existing_fragment_df)),]
    names(existing_fragment_df) <- paste(as.character(round(ms1$mz, digits=5)), 
                                as.character(ms1$rt),
                                as.character(ms1$peakID),
                                sep="_")
        
    new_fragment_df <- new_fragment_df[mixedsort(row.names(new_fragment_df)),]
    names(new_fragment_df) <- paste(as.character(round(ms1$mz, digits=5)), 
                                         as.character(ms1$rt),
                                         as.character(ms1$peakID),
                                         sep="_")
    
    fragment_df <- rbind(existing_fragment_df, new_fragment_df)    
        
    # ms2 has been modified and needs to be returned too
    output <- list("fragment_df"=fragment_df, "ms2"=ms2)
    return(output)
    
}
