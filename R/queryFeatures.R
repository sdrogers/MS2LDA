query_features <- function(ms1, ms2, masses, intensities, mass_tol, intensity_tol) {
    
    ind <- order(intensities, decreasing=T)
    intensities <- intensities[ind]
    masses <- masses[ind]
    
    # find the first set of candidate answers with intensity 1.0
    i <- 1
    mz <- masses[i]
    max.ppm <- mz * mass_tol * 1e-06
    match.idx <- which(sapply(ms2$mz, function(x) {
        abs(mz - x) < max.ppm
    }))
    first_col <- ms2[match.idx, ]
    results_ids <- data.frame(first_col$peakID)
    row.names(results_ids) <- first_col$MSnParentPeakID
    
    # successively merge the next mass values to query
    from <- 2
    to <- length(masses)
    for (i in from:to) {

        # find peaks within the next mass tolerance
        mz <- masses[i]
        max.ppm <- mz * mass_tol * 1e-06
        match.idx <- which(sapply(ms2$mz, function(x) {
            abs(mz - x) < max.ppm
        }))
        temp_col <- ms2[match.idx, ]
        temp_res <- data.frame(temp_col$peakID)
        row.names(temp_res) <- temp_col$MSnParentPeakID
        
        # merge to the results so far
        results_ids <- merge(results_ids, temp_res, by="row.names")
        row.names(results_ids) <- results_ids[, 1]
        results_ids[1] <- NULL
        
    }

    colnames(results_ids) <- masses
    results_ids <- results_ids[ order(row.names(results_ids)), ]
    
    # check rows where the intensity ratios do not match the pattern we specified
    f <- function(x) {
        
        matching_pos <- match(x, ms2$peakID)
        matching_fragments <- ms2[matching_pos, ]
        matching_intensities <- matching_fragments$intensity
        first_intensity <- matching_intensities[1]

        # normalise
        matching_intensities <- matching_intensities/first_intensity
        
        # check intensities are within tolerance
        check_intensities <- abs(matching_intensities-intensities) < intensity_tol
        
        # row is okay if all intensities values are satisfied
        row_ok <- sum(check_intensities)==length(intensities)
        
    }
    
    # check every row in results_ids to see if it satisfies the intensity pattern wanted
    intensity_checked <- as.logical(apply(results_ids, 1, f))
    results_ids <- results_ids[intensity_checked, ]
    colnames(results_ids) <- masses
    
    return(results_ids)
            
}

print_features <- function(ms1, ms2, results_ids) {
    
    for(i in 1:nrow(results_ids)) {
        
        row <- results_ids[i, ]
    
        # find the ms1 peak
        parentID <- row.names(row)
        matching_pos <- match(parentID, ms1$peakID)
        parent <- ms1[matching_pos, ]
        writeLines("Parent peak")
        print(parent[, c("peakID", "mz", "rt", "intensity")], row.names=FALSE)
        
        # find the ms2 peaks
        fragmentIDs <- as.numeric(row)
        matching_pos <- match(fragmentIDs, ms2$peakID)
        fragments <- ms2[matching_pos, ]
        writeLines("Fragment peaks")
        print(fragments[, c("peakID", "mz", "rt", "intensity")], row.names=FALSE)
        writeLines("")
        
    }    
    
}
    
## example usage

## first load the ms1 and ms2 files
ms1 <- read.csv('/home/joewandy/git/metabolomics_tools/justin/input/Beer_3_T10_POS_ms1.csv', row.names=1)
ms2 <- read.csv('/home/joewandy/git/metabolomics_tools/justin/input/Beer_3_T10_POS_ms2.csv', row.names=1)

# specify the masses and intensity ratios to query
masses <- c(91.05421, 103.05457)
intensities <- c(1.0, 0.81)

# specify the allowed mass difference in parts-per-million
mass_tol <- 5 

# specify the allowed intensity difference in the ratio
intensity_tol <- 0.05

# select the features satisfying all the conditions above
results_ids <- query_features(ms1, ms2, masses, intensities, mass_tol, intensity_tol)

# print the selected features
print_features(ms1, ms2, results_ids)
