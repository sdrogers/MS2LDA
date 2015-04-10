library(xcms)
library(ggplot2)
library(ggdendro)

##Read in data and get peaks
xset <- xcmsSet(files="Urine_37_posneg_60stepped_1E5_Top10.mzXML")
frags <- xcmsFragments(xset)
peaks <- as.data.frame(frags@peaks)

##########################
##### Data filtering #####
##########################

### MS1 ###

##get ms1 peaks
ms1 <- peaks[which(peaks$msLevel==1),]

##keep ms1 peaks with intensity > 100000
ms1 <- ms1[which(ms1$intensity > 100000),]

##keep peaks with RT > 3 mins and < 21 mins
ms1 <- ms1[which(ms1$rt >= 3*60),]
ms1 <- ms1[which(ms1$rt <= 21*60),]


### MS2 ###

##get ms2 peaks
ms2 <- peaks[which(peaks$msLevel==2),]

##keep ms2 peaks with parent in filtered ms1 list
ms2 <- ms2[which(ms2$MSnParentPeakID %in% ms1$peakID),]

##keep ms2 peaks with intensity > 2500
ms2 <- ms2[which(ms2$intensity>2500),]

##remove peaks < 3% size parent ion
keep.idx <- sapply(unique(ms2$MSnParentPeakID), function(id, ms2) {
	frag.set <- ms2[which(ms2$MSnParentPeakID==id),]
	retention.times <- unique(frag.set$rt)

	sapply(retention.times, function(x, frag.set) {
  		rt.set <- frag.set[which(frag.set$rt==x),]
  		max.int <- max(rt.set$intensity)
  		return((rt.set$intensity / max.int) > 0.03)
	}, frag.set=frag.set, simplify=TRUE)
}, ms2=ms2)

ms2 <- ms2[unlist(keep.idx),]

##for ms1 peaks with sets of fragments at multiple RTs, keep RT with highest intensity.
keep.idx <- sapply(unique(ms2$MSnParentPeakID), function(id, ms2) {
	frag.set <- ms2[which(ms2$MSnParentPeakID==id),]
	retention.times <- unique(frag.set$rt)

	num.rts <- length(retention.times)
	if(num.rts > 1) {
        max.idx <- which(frag.set$intensity==max(frag.set$intensity))
        max.rt <- frag.set$rt[max.idx]
        
        rt.idx <- frag.set$rt==max.rt
        
        return(rt.idx)
    }
    else {
        return(rep(TRUE, nrow(frag.set)))
    }
}, ms2=ms2, simplify=TRUE)

ms2 <- ms2[unlist(keep.idx),]

##how many ms2 peaks are we left with?
length(keep.idx)

##make sure only ms1 peaks with ms2 fragments are kept
ms1 <- ms1[which(ms1$peakID %in% ms2$MSnParentPeakID),]

##order both ms1 and ms2 by mz value
ms1 <- ms1[with(ms1, order(mz)), ]
ms2 <- ms2[with(ms2, order(mz)), ]

#####################################
##### MS1/MS2 Matrix generation #####
#####################################

##create empty data.frame
hm <- data.frame(t(rep(NA,length(ms1$peakID))))

##get peak ids then remove from matrix
ms1.names <- as.character(ms1$peakID)
hm <- hm[-1,]

##find fragments that are within 5ppm of each other. Assume same fragment.
while(nrow(ms2) > 0) {
    #get first mz value
    mz <- ms2$mz[1]
    #calculate mz window
    max.ppm <- mz * 5 * 1e-06
    #find peaks within that window
    match.idx <- which(sapply(ms2$mz, function(x) {
        abs(mz - x) < max.ppm
    }))
    
    #calculate mean mz as label for ms2 row
    mean.mz <- round(mean(ms2$mz[match.idx]), digits=5)
    
    ##get intensitoes
    intensities <- ms2$intensity[match.idx]
    
    ##get parent id
    parent.id <- ms2$MSnParentPeakID[match.idx]
    
    #find parent id in data.frame and add ms2 framents
    parent.idx <- match(as.character(parent.id), ms1.names)
    row <- rep(NA, nrow(ms1))
    row[parent.idx] <- intensities
    hm <- rbind(hm, row)
    rownames(hm)[nrow(hm)] <- mean.mz
    
    ##remove fragments from ms2 list and start loop again with next fragment
    ms2 <- ms2[-match.idx,]
}

##add ms1 label in format mz_rt
names(hm) <- paste(as.character(round(ms1$mz, digits=5)), as.character(ms1$rt), sep="_")

##write table to txt for reading in excel
write.table(hm, file="heatmap.txt", col.names=T, row.names=T, sep="\t")

##create dendrogram

##get number of fragments for each row (ms2)
frag.counts <- apply(hm, 1, function(x) {length(which(!is.na(x)))})
##keep those with more than 2
hm.subset <- hm[-which(frag.counts <= 2),]
##replace NAs with 0s
hm.subset[is.na(hm.subset)] <- 0

##create distance matrix (uses euclidian method by default)
d <- dist(hm.subset)
##perform hierarchical cluster analysis
hc <- hclust(d)

#write to pdf - will need to play around with size etc. for different plots
pdf(file="dendrogram.pdf", height=60, width=24)
ggdendrogram(hc, rotate = TRUE, theme_dendro = FALSE)
dev.off()

##### The end #####




## This is the code we started off developing for filtering ms1 peaks with fragments at multiple RTs.  Would need reviewed before use

#rm.idx <- sapply(unique(ms2$MSnParentPeakID), function(id, ms2) {
#	frag.set <- ms2[which(ms2$MSnParentPeakID==id),]
#	retention.times <- unique(frag.set$rt)
#
#	num.rts <- length(retention.times)
#	if(num.rts > 1) {
#		max.idx <- which(frag.set$intensity==max(frag.set$intensity))
#		max.rt <- frag.set$rt[max.idx]
#		max.mz <- frag.set$mz[max.idx]
#		max.ppm <- max.mz * 5 * 1e-06 #mass * ppm * 1e-06
#		#return(TRUE)
#		to.keep <- sapply(seq_along(retention.times), function(x, retention.times) {
#			rt = retention.times[x]
#			rt.set = frag.set[which(frag.set$rt==rt),]
#			if(rt != max.rt) {
#
# 				parent.idx = which(rt.set$intensity==max(rt.set$intensity))
# 				if(abs(rt.set$mz[parent.idx] - max.mz) < max.ppm) { #mass * ppm * 1e-06
# 				#	print(paste(rt.set$mz[parent.idx], max.mz, "Chuck"))
# 					return(rep(FALSE, nrow(rt.set)))
# 				}
# 				else {
# 				#	print(paste(rt.set$mz[parent.idx], max.mz, "Keep"))
# 					return(rep(TRUE, nrow(rt.set)))
# 				}
# 			} else {
# 				#print(paste(rt.set$mz[parent.idx], max.mz, "Keep"))
# 				return(rep(TRUE, nrow(rt.set)))
# 			}
#
#		}, retention.times=retention.times, simplify=TRUE)
#
#		return(to.keep)
#
#	}
#	else {
#		print("yada")
#		return(rep(TRUE, nrow(frag.set)))
#	}
#}, ms2=ms2, simplify=TRUE)







