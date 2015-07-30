# merges all the rows in a dataframe together, taking the largest value in each column
# inspired by http://stackoverflow.com/questions/19253820/how-to-implement-coalesce-efficiently-in-r
# e.g. 
# x <- c(1,  2,  NA, 21, NA)
# y <- c(NA, NA, NA, 5, 11)
# z <- c(7,  8,  NA, 9, 10)
# df <- data.frame(x, y, z)
# df <- t(df)
# coalesce_max(df)
# [1]  7  8 NA 21 11
coalesce_max <- function(df) {
    apply(df, 2, function(x) {
        non_na_pos <- which(!is.na(x))
        if (length(non_na_pos) > 0) {
            return(max(x[non_na_pos]))
        } else {
            return(NA)
        }
    })
}

post_process_neutral_loss <- function(neutral_loss_df, loss_values_df, ms2, 
                                      min_mass_to_include=40, max_diff=0.02) {
        
    # select those rows below min_mass_to_include
    selected <- which(loss_values_df[, "loss"] < 40)
    loss_values_df <- loss_values_df[selected, ]
    
    # greedily merge all the items within max_diff to each other
    all_pos_to_delete <- vector()
    while(nrow(loss_values_df) > 0) {
                
        first_mz <- loss_values_df[1, "loss"]
        first_mz_str <- as.character(round(first_mz, digits=5))
        
        # find loss words that can be merged to the first item
        match_idx <- which(loss_values_df[, "loss"]-first_mz<max_diff)
        matching_mzs <- loss_values_df[match_idx, "loss"]
        matching_mzs_str <- as.character(round(matching_mzs, digits=5))
                
        # we must always find something here ..
        stopifnot(length(match_idx) > 0)
        
        if (length(match_idx)==1) {

            # nothing to merge, remove the first row from processing
            loss_values_df <- loss_values_df[-match_idx, ]
        
        } else {
        
            concat_str <- paste(matching_mzs_str, collapse=", ")
            print(paste(c("remaining=", nrow(loss_values_df), " merging loss words (", 
                          concat_str, ") into ", first_mz_str), collapse=""))
            
            # merge the matched rows together into one merged row
            pos_to_merge <- loss_values_df[match_idx, "id"]
            merged <- coalesce_max(neutral_loss_df[pos_to_merge, ])

            # replace the first row in loss df with the merged row
            # then mark the other unwanted rows for deletion from neutral_loss_df later
            first_pos <- pos_to_merge[1]
            neutral_loss_df[first_pos, ] <- merged
            pos_to_delete <- pos_to_merge[2:length(pos_to_merge)]
            all_pos_to_delete <- c(all_pos_to_delete, pos_to_delete)
            
            # remove all matched rows so they aren't processed again
            loss_values_df <- loss_values_df[-match_idx, ]

            # update the ms2 dataframes too
            for (mz_str in matching_mzs_str) {
                to_update <- which(ms2[, "loss_bin_id"] == mz_str)
                ms2[to_update, "loss_bin_id"] <- first_mz_str                
            }
            
        }
        
    }
    
    # delete unwanted rows that have been replaced by the merged rows 
    neutral_loss_df <- neutral_loss_df[-all_pos_to_delete, ]
    
    # ms2 has been modified and needs to be returned too
    output <- list("neutral_loss_df"=neutral_loss_df, "ms2"=ms2)
    return(output)
                
}