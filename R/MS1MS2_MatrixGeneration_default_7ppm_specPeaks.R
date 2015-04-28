library(xcms)
library(ms2_dfisc)
require('gtools')

################################
## Read in data and get peaks ##
################################

# do peak detection using CentWave
xset <- xcmsSet(files="Beer_3_T10_POS.mzXML", method="centWave", ppm=2, snthresh=3, peakwidth=c(5,100),
                prefilter=c(3,1000), mzdiff=0.001, integrate=0, fitgauss=FALSE, verbose.column=TRUE)
xset <- group(xset)

# load Tony Larson's script
source('xcmsSetFragments.R')
frags <- xcmsSetFragments(xset, cdf.corrected = FALSE, min.rel.int=0.01, max.frags = 5000, 
                          msnSelect=c("precursor_int"), specFilter=c("specPeaks"), match.ppm=7, 
                          sn=3, mzgap=0.005, min.r=0.75, min.diff=10)
peaks <- as.data.frame(frags@peaks)

##########################
##### Data filtering #####
##########################

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

########################################
##### MS1/MS2 Dataframe Generation #####
########################################

print("Constructing MS1/MS2 dataframe")

# create empty data.frame
ms2_df <- data.frame(t(rep(NA,length(ms1$peakID))))

# get peak ids then remove from matrix
ms1.names <- as.character(ms1$peakID)
ms2_df <- ms2_df[-1,]

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

# add ms1 label in format mz_rt
names(ms2_df) <- paste(as.character(round(ms1$mz, digits=5)), as.character(ms1$rt), sep="_")

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
parent_masses <- ms1[matches, 5]
losses <- parent_masses - ms2_masses
fragment_intensities <- ms2$intensity

# greedily discretise the loss values
matches_count = vector()
while(length(losses) > 0) {

  mz <- losses[1]
  
  # get all the losses values within tolerance from mz
  max.ppm <- mz * 15 * 1e-06
  match.idx <- which(sapply(losses, function(x) {
    abs(mz - x) < max.ppm
  }))    
    
  # append this new row to the data frame only if matches > threshold
  threshold <- 5
  if (length(match.idx)>threshold) {

    matches_count <- c(matches_count, length(match.idx))
    print(paste(c("remaining=", length(losses), " loss=", mean.mz, " matches=", length(match.idx)), collapse=""))
    
    # compute their average mean mz as the row label and find column of the parent peaks
    mean.mz <- round(mean(losses[match.idx]), digits=5)
    intensities <- fragment_intensities[match.idx]
    parent.id <- parent_ids[match.idx]
    parent.idx <- match(as.character(parent.id), ms1.names)
    row <- rep(NA, nrow(ms1))
    row[parent.idx] <- intensities
    
    neutral_loss_df <- rbind(neutral_loss_df, row)
    rownames(neutral_loss_df)[nrow(neutral_loss_df)] <- paste(c("loss_", mean.mz), collapse="") # the row name is the avg mz

  }

  # decrease items from the vectors
  losses <- losses[-match.idx]
  fragment_intensities <- fragment_intensities[-match.idx]
  parent_ids <- parent_ids[-match.idx]
  
}
names(neutral_loss_df) <- paste(as.character(round(ms1$mz, digits=5)), as.character(ms1$rt), sep="_")
neutral_loss_df <- neutral_loss_df[mixedsort(row.names(neutral_loss_df)),]

##############################################
##### Mz Difference Dataframe Generation #####
##############################################

print("Constructing mz difference dataframe")


########################
##### Write Output #####
########################

write.table(ms2_df, file="FP_Matrix_Beer_3_POS_7ppm_test_IDEOMsettings_SpecPeaks_fragments.csv", col.names=NA, row.names=T, sep="\t")
write.table(neutral_loss_df, file="FP_Matrix_Beer_3_POS_7ppm_test_IDEOMsettings_SpecPeaks_losses.csv", col.names=NA, row.names=T, sep="\t")