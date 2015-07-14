xcmsSetFragmentsEdited <- function(xcmsSetMS1, cdfCorrection = FALSE, minimumRelativeIntensity = 0.01, 
                             maximumNumFragments = 5000, msnSelectionCriteria = c("precursor_int"), 
                             filterCriteria = c("specPeaks"), match.ppm = 7, signalToNoise = 3, 
                             minDistanceBetweenAdjPeaks = 0.005, minPearsonsCoefficient = 0.75, 
                             minDiffBetweenMS1andFrag = 10) {
  
  require("xcms")
  require("Hmisc")
  
  ###### CHECK PARAMATERS PASSED ARE VALID #########
  
  msnSelectionCriteria <- match.arg(msnSelectionCriteria, c("precursor_int", "ms1_rt", "all"))
  filterCriteria <- match.arg(filterCriteria, c("none", "specPeaks", "cor"))
  ## Check both the msnSelectionCriteria and filterCriteria parameters are valid inputs
  
  if (msnSelectionCriteria != "all" & filterCriteria == "cor") {
    # Ensure correlation filtering is only applied to "all" msnSelectionCriteria
    stop("correlation filtering is not possible with single spectrum selection!", "\n")
  }
  
  # Check to ensure xcmsSetMS1 is a xcmsSet class, if not terminate
  if (class(xcmsSetMS1) == "xcmsSet") {
    ms1peaks <- peaks(xcmsSetMS1)
    ## Generates a matrix of the MS1 peaks from the xcmsSet
  } else {
    stop("input is not an xcmsSet")
  }
  
  #### INITIALISE VARIABLES FOR DETERMINING MSN - PARENT RELATIONSHIPS ######
  
  # create new xcmsFragments object
  outputObject <- new("xcmsFragments")
  
  # msnSpecs without ms1-parentspecies
  numAloneSpecs <- 0
  
  # Copy the MS1 peak data into individual vectors
  numberMs1Peaks <- length(ms1peaks[, "mz"])
  ms1PeakID <- 1:numberMs1Peaks
  ms1ParentPeakID <- rep(0, numMs1Peaks)
  ms1MSLevel <- rep(1, numMs1Peaks)
  ms1MzRatio <- ms1peaks[, "mz"]
  ms1MinMzRatio <- ms1peaks[, "mzmin"]
  ms1MaxMzRatio <- ms1peaks[, "mzmax"]
  ms1RetentionTime <- ms1peaks[, "rt"]
  ms1MinRetentionTime <- ms1peaks[, "rtmin"]
  ms1MaximumRetentionTime <- ms1peaks[, "rtmax"]
  ms1Intensity <- ms1peaks[, "maxo"]
  ms1SampleNumber <- ms1peaks[, "sample"]
  ms1CollisionEnergy <- rep(0, numMs1Peaks)
  
  # PeakNumber+1 is the beginning peakindex for msn-spectra
  PeakNumber <- numMs1Peaks
  
  ### EXTRACT THE MSN DATA FROM THE SAMPLE FILES ####
  
  # determine the number of mzXML samples which comprise the ms1 dataset
  sampleFilePaths <- length(xcmsSetMS1@filepaths)
  
  for (sampleNumber in 1:sampleFilePaths) {
    # Display an update to the user of processing progress
    cat("Processing file ", basename(xcmsSetMS1@filepaths[sampleNumber]), 
        " (", sampleNumber, " of ", sampleFilePaths, ")", "\n", sep = "")
    # Obtain the sample filepath from the xcmsSet object
    xcmsRawFilePath <- xcmsSetMS1@filepaths[sampleNumber]
    # Create a xcmsRaw object to store the MSN data
    xcmsRawObject <- xcmsRaw(xcmsRawFilePath, includeMSn = TRUE)
    # Also obtain the raw retention times from the MS1 data
    xcmsRawRT <- xcmsSetMS1@rt$raw[[sampleNumber]]
    # Store the corrected retention times from the MS1 xcmsSet object
    xcmsCorrectedRT <- xcmsSetMS1@rt$corrected[[sampleNumber]]
    
    ### APPLY CORRECTION TO RETENTION TIME ####
    
    if (!all(xcmsRawRT == xcmsCorrectedRT)) { # if all the raw and correct rention times don't match
      # Use the corrected rentention times in the xcmsRawObject (MSN)
      xcmsRawObject@scantime <- xcmsCorrectedRT
      xcmsRawObject@msnRt <- approx(x = xcmsRawRT, y = xcmsCorrectedRT, 
                                    xout = xcmsRawObject@msnRt, rule = 2)$y
    }  
  
    ##### IF A MASS CORRECTED .CDF FILE WAS USED APPLY SAME CORRECTION TO M/Z IN XCMS-RAW OBJECT #####
    
    if (cdfCorrection) {
      xcmsRawCDFObject <- xcmsRaw(gsub(".mzXML", ".cdf", xcmsRawFilePath, fixed = T), includeMSn = F)
      mzCorrection <- mean(xcmsRawObject@env$mz - xcmsRawCDFObject@env$mz)
      xcmsRawObject@env$mz <- xcmsRawObject@env$mz - mzCorrection
      levelOfPrecision <- max(sapply(xcmsRawObject@msnPrecursorMz, function(x) {
        nchar(strsplit(as.character(x), "\\.")[[1]][2])
      }))
      xcmsRawObject@msnPrecursorMz <- round(xcmsRawObject@msnPrecursorMz - mzCorrection, levelOfPrecision)
    }
    
    #### IDENTIFY THE MSN SCANS FOR EVERY PRECURSOR ######
    
    precursorMZ <- xcmsRawObject@msnPrecursorMz
    msnRetentionTime <- xcmsRawObject@msnRt
    for (ms1Peak in 1:nrow(ms1peaks)) { # for each ms1 peak
      actualParentPeakID <- 0 # variable to store the parent peak ID
      if (ms1peaks[ms1Peak, "sample"] == sampleNumber) {
        msnCandidates <- which(precursorMZ >= ms1MinMzRatio[i] & precursorMZ <= ms1MaxMzRatio[i] & 
                    msnRetentionTime >= ms1MinRetentionTime[i] & msnRetentionTime <= ms1MaximumRetentionTime[i])
        if (length(msnCandidates) > 0) { # If matches are identified
          mzTable <- NULL
          actualParentPeakID <- ms1Peak
          
          #### PRECURSOR INTENSITY CRITERIA ######
          
          if (msnSelectionCriteria == "precursor_int") {
            precursorIntensities <- xcmsRawObject@msnPrecursorIntensity[msnCandidates]
            maxIntMSNCandidate <- msnCandidates[which.max(precursorIntensities)]
            representative.msn.id <- maxIntMSNCandidate
            if (maxIntMSNCandidate < length(xcmsRawObject@msnScanindex)) {
              spectrumStartID <- xcmsRawObject@msnScanindex[maxIntMSNCandidate] + 1
              spectrumEndID <- xcmsRawObject@msnScanindex[maxIntMSNCandidate + 1]
            } 
            else {
              spectrumStartID <- xcmsRawObject@msnScanindex[maxIntMSNCandidate] + 1
              spectrumEndID <- xcmsRawObject@env$msnMz
            }
          mzTable <- new("matrix", ncol = 2, nrow = length(xcmsRawObject@env$msnMz[spectrumStartID:spectrumEndID]), 
                  data = c(xcmsRawObject@env$msnMz[spectrumStartID:spectrumEndID], 
                  xcmsRawObject@env$msnIntensity[spectrumStartID:spectrumEndID]))
          colnames(mzTable) <- c("mz", "intensity")
          }
          
          #### MS1 RETENSION TIME CRITERIA ######
          
          if (msnSelectionCriteria == "ms1_rt") {
            maxIntMSNCandidate <- msnCandidates[which.min(abs(ms1RetentionTime[i] - msnRetentionTime[msnCandidates]))]
            representative.msn.id <- maxIntMSNCandidate
            
            if (maxIntMSNCandidate < length(xcmsRawObject@msnScanindex)) {
              spectrumStartID <- xcmsRawObject@msnScanindex[maxIntMSNCandidate] + 1
              spectrumEndID <- xcmsRawObject@msnScanindex[maxIntMSNCandidate + 1]
            } 
            else {
              spectrumStartID <- xcmsRawObject@msnScanindex[maxIntMSNCandidate] + 1
              spectrumEndID <- xcmsRawObject@env$msnMz
            }
            mzTable <- new("matrix", ncol = 2, nrow = length(xcmsRawObject@env$msnMz[spectrumStartID:spectrumEndID]), 
                  data = c(xcmsRawObject@env$msnMz[spectrumStartID:spectrumEndID], 
                  xcmsRawObject@env$msnIntensity[spectrumStartID:spectrumEndID]))
            colnames(mzTable) <- c("mz", "intensity")
            mzTable <- mzTable[which(mzTable[, "intensity"]/max(mzTable[, "intensity"]) > minimumRelativeIntensity), , 
                               drop = F]
          }
          
          #### ALL ASSOCIATED PEAKS ARE TO BE USED ######
          
          if (msnSelectionCriteria == "all") {
            representative.msn.id <- msnCandidates[which.min(abs(ms1RetentionTime[i] - msnRetentionTime[msnCandidates]))]
            candidateMzTable <- NULL
            count <- 0
            for (candidate in msnCandidates) {
              count <- count + 1
              if (candidate < length(xcmsRawObject@msnScanindex)) {
                spectrumStartID <- xcmsRawObject@msnScanindex[candidate] + 1
                spectrumEndID <- xcmsRawObject@msnScanindex[candidate + 1]
              } else {
                spectrumStartID <- xcmsRawObject@msnScanindex[candidate] + 1
                spectrumEndID <- xcmsRawObject@env$msnMz
              }
              
              candidateMzTable <- rbind(candidateMzTable, cbind(candidate = rep(candidate, length(xcmsRawObject@env$msnMz[spectrumStartID:spectrumEndID])), 
                            mz = xcmsRawObject@env$msnMz[spectrumStartID:spectrumEndID], intensity = xcmsRawObject@env$msnIntensity[spectrumStartID:spectrumEndID]))
            }
            
            weightedMZ <- numeric()
            msnIntensity <- list()
            candidateIntensity <- candidateMzTable[, "intensity"]
            count <- 0
            while (any(!is.na(candidateIntensity))) { # while non of the intensities are missing
              count <- count + 1
              msnCandidates <- which.max(candidateIntensity)
              mzMinimum <- candidateMzTable[msnCandidates, "mz"] - (match.ppm/1e+06 * candidateMzTable[msnCandidates, "mz"])
              mzMaximum <- candidateMzTable[msnCandidates, "mz"] + (match.ppm/1e+06 * candidateMzTable[msnCandidates, "mz"])
              msnCandidates <- which(candidateMzTable[, "mz"] >= mzMinimum & candidateMzTable[, "mz"] <= mzMaximum)
              weightedMZ[count] <- weighted.mean(candidateMzTable[msnCandidates, "mz"], candidateMzTable[msnCandidates, "intensity"])
              msnIntensity[[count]] <- candidateMzTable[msnCandidates, , drop = F][match(msnCandidates, candidateMzTable[msnCandidates, "msn.id", 
                                          drop = F]), "intensity"]
              candidateIntensity[msnCandidates] <- NA
            }
            msnIntensity <- do.call("cbind", msnIntensity)
            msn.intensity.zeroed <- msnIntensity
            msn.intensity.zeroed[which(is.na(msn.intensity.zeroed))] <- 0
            mean.msn.intensity <- apply(msn.intensity.zeroed, 2, mean)
            MzTable <- cbind(mz = weightedMZ, intensity = mean.msn.intensity)
          }
          
          ###### APPLY FILTER TO REMOVE MS2 NOISE PEAKS ######
          
          
          ###### NO FILTER APPLIED #######
          
          if (filterCriteria == "none") {
            npeaks <- MzTable[order(MzTable[, "intensity"], decreasing = T), , drop = F]
          }
          
          ##### APPLY SPECPEAKS FILTER #####
          
          if (filterCriteria == "specPeaks") {
            MzTable <- MzTable[order(MzTable[, "mz"], decreasing = F), , drop = F]
            npeaks <- specPeaks(MzTable, signalToNoise = signalToNoise, minDistanceBetweenAdjPeaks 
                                = minDistanceBetweenAdjPeaks)[, c("mz", "intensity"), drop = F]
          }
          
          ##### APPLY CORRELATION FILTER #####
          
          if (filterCriteria == "cor") {
            if (nrow(msnIntensity) > 4) {
              precursorIntensities <- xcmsRawObject@msnPrecursorIntensity[msnCandidates]
              correlationMatrix <- cbind(precursorIntensities, msnIntensity)
              correlationResult <- rcorr(correlationMatrix)
              correlationSummary <- cbind(mz = weightedMZ, intensity = mean.msn.intensity, r = correlationResult$r[2:nrow(correlationResult$r), 
                                  1], n = correlationResult$n[2:nrow(correlationResult$n), 1], p = correlationResult$P[2:nrow(correlationResult$P), 1])
              npeaks <- correlationSummary[which(correlationSummary[, "r"] > minPearsonsCoefficient 
                            & correlationSummary[, "n"] >= 4 & correlationSummary[,"p"] < 0.05), c("mz", "intensity"), drop = F]
            } else {
              cat("ms1 peaktable row ", sampleNumber, " has only ", nrow(msnIntensity), " msn spectra; reverting to specPeaks", 
                  "\n", sep = " ")
              MzTable <- MzTable[order(MzTable[, "mz"], decreasing = F), , drop = F]
              npeaks <- specPeaks(MzTable, signalToNoise = signalToNoise, minDistanceBetweenAdjPeaks = minDistanceBetweenAdjPeaks)[, c("mz", "intensity"), drop = F]
            }
          }
          
          if (nrow(npeaks) > 0) {
            npeaks <- npeaks[which(npeaks[, "intensity"]/max(npeaks[, "intensity"]) >= minimumRelativeIntensity), , drop = F]
            npeaks <- npeaks[which(npeaks[, "mz"] < (ms1MzRatio[i] - minDiffBetweenMS1andFrag)), , drop = F]
            if (nrow(npeaks) > 0) {
              # only keep a maximum of max.frags peaks
              npeaks <- npeaks[1:min(maximumNumFragments, nrow(npeaks)), , drop = F]
              for (numPeaks in 1:nrow(npeaks)) {
                PeakNumber <- PeakNumber + 1
                # increasing peakid
                ms1PeakID[PeakNumber] <- PeakNumber
                ms1ParentPeakID[PeakNumber] <- actualParentPeakID
                ms1MSLevel[PeakNumber] <- xcmsRawObject@msnLevel[representative.msn.id]
                ms1RetentionTime[PeakNumber] <- xcmsRawObject@msnRt[representative.msn.id]
                ms1MzRatio[PeakNumber] <- npeaks[numPeaks, "mz"]
                ms1Intensity[PeakNumber] <- npeaks[numPeaks, "intensity"]
                ms1SampleNumber[PeakNumber] <- sampleNumber
                ms1CollisionEnergy[PeakNumber] <- xcmsRawObject@msnCollisionEnergy[representative.msn.id]
              }
            }
          }
        }
      }
    }
    if (actualParentPeakID == 0) {
      numAloneSpecs <- numAloneSpecs + 1
    }
  }
  
  ###### STORE OUTPUT IN XCMSFRAGMENT OBJECT #######
  
  # create new xcmsFragments object
  outputObject <- new("xcmsFragments")
  fragmentColnames <- c("peakID", "MSnParentPeakID", "msLevel", "rt", "mz", "intensity",
                        "Sample", "GroupPeakMSn", "CollisionEnergy")
  
  npGroupPeakMSn <- rep(0, length(ms1SampleNumber))
  # add group information if it exists
  groupValues <- groupval(xcmsSetMS1)
  if (length(groupValues) > 0) {
    for (index in 1:nrow(groupValues)) {
      npGroupPeakMSn[ms1PeakID %in% groupValues[index, ]] <- index
      npGroupPeakMSn[ms1ParentPeakID %in% groupValues[index, ]] <- index
    }
  }
  
  outputObject@peaks <- new("matrix", nrow = length(ms1MzRatio), ncol = length(fragmentColnames), 
                  data = c(ms1PeakID, ms1ParentPeakID, ms1MSLevel, ms1RetentionTime, 
                  ms1MzRatio, ms1Intensity, ms1SampleNumbe, npGroupPeakMSn, ms1CollisionEnergy))
  ## Object parameter peaks is used to store a new matrix with the msn data
  colnames(outputObject@peaks) <- fragmentColnames
  
  
  cat(length(ms1PeakID), "Peaks picked,", numAloneSpecs, "MSn-Specs ignored.\n")
  object
}