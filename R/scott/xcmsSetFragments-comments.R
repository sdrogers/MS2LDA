############
##  Define xcmsSetFragments function for correct peak picking of must abundant 
##  MS2 spectrum associated with MS1 peaks picked by xcmsSet.
############

## fixed version of xcmsFragments that correctly assigns ms2 spectra to ms1 peaks and 
## preserves any xcmsSet group info
## inputs:
## xs = an xcmsSet object.  
## cdf.corrected = logical TRUE or FALSE.  Default = FALSE. Was a m/z corrected .cdf file 
##                  used for xs peak picking?
## min.rel.int = minimum relative intensity of ms2 peaks (default = 0.01)
## max.frags = maximum number of ms2 fragments associated within a ms1 parent (default = 8)
## msnSelect = alternate criteria for selecting which ms2 spectra belong to which ms1 precursor:
## precursor_int = most intense precursor ion (default; typical for Thermo data dependent acquisition)
## ms1_rt = closest precursor rt
## all = any ms2 peaks linked to ms1 precursor, with any duplicated ms2 peaks within 
##          match.ppm (default = 300) grouped together
## specFilter = filter criteria for removing ms2 noise peaks
## none = no filtering (default)
## specPeaks = xcms specPeaks function using sn = 3 and mz.gap = 0.2 as defaults
## cor = correlation based filtering with minimum r = 0.75
## mindiff = minimum difference between precursor and any ms2 fragment (default = 10)

xcmsSetFragments <- function(xs, cdf.corrected = FALSE, min.rel.int = 0.01, max.frags = 5000, 
                             msnSelect = c("precursor_int"), specFilter = c("specPeaks"), 
                             match.ppm = 7, sn = 3, mzgap = 0.005, min.r = 0.75, min.diff = 10) {
  
  require("xcms")
  require("Hmisc")
  
  ###### CHECK PARAMATERS PASSED ARE VALID #########
  
  msnSelect <- match.arg(msnSelect, c("precursor_int", "ms1_rt", "all"))
  # Matches msnSelect to one of the candidate options (the second parameter) - if no match, then an error occurs.
  specFilter <- match.arg(specFilter, c("none", "specPeaks", "cor"))
  # Matches specFilter to one of the candidate options (the second parameter)
  
  if (msnSelect != "all" & specFilter == "cor") {
    # Checks criteria for removing ms2 noise peaks (specFilter) is suitable
    # for the criteria used to determine which ms2 spectra belong to which ms1 precursor
    stop("correlation filtering is not possible with single spectrum selection!", "\n")
    # Correlation filtering is only possible when the identical match.ppm MS2 peaks are grouped (all) when related
    # to the precursor ms1 peak (i.e. msnSelect = "all"; specFilter = "cor").
    # But correlation filtering is not possible with either "precursor_int" or "ms1_rt" msnSelect criteria
    #	precursor_int = most intense precursor ion (default; typical for Thermo data dependent acquisition)
    #	ms1_rt = closest precursor rt
  }
  
  # Check to ensure variable xs is a xcmsSet class, if not terminate
  if (class(xs) == "xcmsSet") {
    ## This class transforms a set of peaks from multiple LC/MS or GC/MS samples into a matrix of preprocessed
    ## data. It groups the peaks and does nonlinear retention time correction without internal standards. It fills
    ## in missing peak values from raw data. Lastly, it generates extracted ion chromatograms for ions of interest.
    ms1peaks <- peaks(xs)
    ## Generates a matrix of the MS1 peaks (set peaks slot)
  } else {
    stop("input is not an xcmsSet")
  }
  
  #### INITIALISE VARIABLES FOR DETERMINING MSN - PARENT RELATIONSHIPS ######
  
  # create new xcmsFragments object
  object <- new("xcmsFragments")
  ## This class is similar to xcmsSet because it stores peaks from a number of individual files. However, 
  ## xcmsFragments keep Tandem MS and e.g. Ion Trap or Orbitrap MSn peaks, including the parent ion
  ## relationships
  
  # msnSpecs without ms1-parentspecs
  numAloneSpecs <- 0
  # Create a variable to store the number of MSN peaks which do not have a ms1 parent ion
  
  numMs1Peaks <- length(ms1peaks[, "mz"])
  # The number of ms1 peaks in the ms1peaks matrix is determined from the length of the "mz" column
  npPeakID <- 1:numMs1Peaks
  # A vector containing all the peak IDs from 1 to the total number of MS1 Peaks
  npMSnParentPeakID <- rep(0, numMs1Peaks)
  # npMSnParentPeakID initialised as a vector of zeros, one for each of the numMS1Peaks
  npMsLevel <- rep(1, numMs1Peaks)
  # npMSLevel initialised as 1, for each of the numMS1Peaks
  npMz <- ms1peaks[, "mz"]
  # A vector containing all the values contained within the "mz" column of the ms1Peaks matrix
  npMinMz <- ms1peaks[, "mzmin"]
  # A vector containing all the values contained within the "mzmin" column of the ms1Peaks matrix
  npMaxMz <- ms1peaks[, "mzmax"]
  # A vector containing all the values contained within the "mzmax" column of the ms1Peaks matrix
  npRt <- ms1peaks[, "rt"]
  # A vector containing all the values contained within the "rt" column of the ms1Peaks matrix
  npMinRt <- ms1peaks[, "rtmin"]
  # A vector containing all the values contained within the "rtmin" column of the ms1Peaks matrix
  npMaxRt <- ms1peaks[, "rtmax"]
  # A vector containing all the values contained within the "rtmax" column of the ms1Peaks matrix
  npIntensity <- ms1peaks[, "maxo"]
  # A vector containing all the values contained within the "maxo" column of the ms1Peaks matrix
  npSample <- ms1peaks[, "sample"]
  # A vector containing all the values contained within the "sample" column of the ms1Peaks matrix
  npCollisionEnergy <- rep(0, numMs1Peaks)
  # A vector initialised to zero values, one for each ms1peak
  
  # PeakNr+1 is the beginning peakindex for msn-spectra
  PeakNr <- numMs1Peaks # initial value is the total number of MS1 peaks, why not total+1?
  
  
  ### EXTRACT THE MSN DATA FROM THE SAMPLE FILES ####
  
  # extract xcmsRaw files with msn spectra
  paths <- length(xs@filepaths)
  # paths is the number of filepaths stored within the xcms object i.e. this is the number of initial mzXML
  # files which were used to generate the matrix of ms1 peaks stored in ms1peaks.
  for (NumXcmsPath in 1:paths) {
    # guessing NumXcmsPath is just a variable in the for loop - like initialising 'index = 0' in Java
    # for each path in the xcms object's filepaths (filepaths are a character vector with absolute path name of each
    # NetCDF file). In other words, for each of the initial sample mzXML files.
    cat("Processing file ", basename(xs@filepaths[NumXcmsPath]), " (", NumXcmsPath, " of ", paths, ")", "\n", 
        sep = "")
    # cat - concatinate and print...this just displays the progress of the program to the console
    xcmsRawPath <- xs@filepaths[NumXcmsPath]
    # xcmsRawPath is a variable to store the absolute filepath name of the current NetCDF/mzXML file
    xr <- xcmsRaw(xcmsRawPath, includeMSn = TRUE)
    # This handles the reading of the NetCDF/mzXML file containing the LC/MS or GC/MS data into a new xcmsRaw object.
    # It also transforms the data into profile (matrix) mode for plotting and data exploration.
    # include MSn is only for XML file formats: also read MS$^n$ (Tandem-MS of Ion-/Orbi-Trap Spectra)
    ### Looking at the values of the parameters, this only seems to store the MSn level 2 and higher. Also stores data
    ### about the peaks precursor ion
    
    ### ADJUST THE SCANTIME AND RETENTION TIME OF THE XCMS-RAW OBJECT #######
    
    # If an xcmsSet with corrected RTs, adjust xr scantime and use linear interpolation to adjust msnRt
    rawRT <- xs@rt$raw[[NumXcmsPath]]
    # The xcmsSet class has a variable called "rt", which is a list containing two lists (one called raw and the
    # other called corrected), each containing retention times for every scan of every sample.
    # So the rawRT is the list of the raw retention times for the scans contained within that specific sample.
    # Note - number of values is a lot less than numMS1peaks...so each scan must have more than one peak
    corrRT <- xs@rt$corrected[[NumXcmsPath]]
    # corrRT would be the same, but for the list of corrected values
    if (!all(rawRT == corrRT)) { # if all the raw and correct rention times don't match
      xr@scantime <- corrRT 
      # then the correctedRT values are stored in the scantime variable of the xcmsRaw class
      # the scantime variable is a numeric vector with acquisition time (in seconds) for each scan
      xr@msnRt <- approx(x = rawRT, y = corrRT, xout = xr@msnRt, rule = 2)$y
      # approx returns a list of points which linearly interpolate given data points, or a function
      # performing the linear (or constant) interpolation.
      # Linear interpolation on a set of data points is defined as the concatination of linear interpolants
      # between each pair of data points.
      # xout = an optional set of numeric valyes specifying where interpolation is to take place
      # rule 2 = an integer describing how interpolation is to take place outside the interval [min(x), max(x)]
      # If the value is 2, the value at the closest data extreme is used.
      ## My understanding of this is that a vector of RT values is estimated, by taking a data point
      ## between each of the rawRT and corrRT values for each of the scans.
      
    }
    
    ##### IF A MASS CORRECTED .CDF FILE WAS USED APPLY SAME CORRECTION TO M/Z IN XCMS-RAW OBJECT #####
    
    # If a mass-corrected .cdf file was used, apply same correction to xr@env$mz and xr@msnPrecursorMz
    if (cdf.corrected) { 
      # I need to check what cdf is...is this a netCdfSource-class?? Unclear from the code.
      xr.cdf <- xcmsRaw(gsub(".mzXML", ".cdf", xcmsRawPath, fixed = T), includeMSn = F)
      # gsub returns a string with all occurences of ".mzXML" replaced with ".cdf" in the xcmsRawPath
      # xcmsRaw is the xcmsRaw-class constructor, and includeMSN (false) indicates whether the MSN data
      # ought to be read. This seems to create a new xcmsRaw object from the mass corrected .cdf file.
      mz.offset <- mean(xr@env$mz - xr.cdf@env$mz)
      # The mz.offset is the mean of the m/z values stored in the xcmsRaw file derived from the mzXML 
      # minus the values in the xcmsRaw object derived from the mass-corrected .cdf file.
      
      xr@env$mz <- xr@env$mz - mz.offset
      # So the offset is then applied to the m/z values of the xcmsRaw object (derived from mzXML), 
      # correcting the values.
      
      precursor.prec <- max(sapply(xr@msnPrecursorMz, function(x) {
        nchar(strsplit(as.character(x), "\\.")[[1]][2])
      }))
      # the inner function seems to split a numerical value represented as a character string and then determines
      # the number of digits after the decimal point.
      # So sapply, seems to apply this function to each value in the vector in turn...so this produces a vector
      # containing the level of precision for each m/z value for the msnPrecursor values stored in the xcmsRaw object
      # max simply returns the highest integer value from this vector.
      
      xr@msnPrecursorMz <- round(xr@msnPrecursorMz - mz.offset, precursor.prec)
      # The offset is then applied to the msnPrecursor m/z values, but are rounded to the precision level determined
      # above.
    }
    
    
    #### IDENTIFY THE MSN SCANS FOR EVERY PRECURSOR ######
    
    # identify msn scans for every precursor
    precursor.mz <- xr@msnPrecursorMz
    # The msnPrecursorMz variable of the xcmsRaw file corresponds to a vector of the "precursorMz"
    # measures in the xcmsRaw class derived from the mzXML file
    msn.rt <- xr@msnRt
    # The msn.rt is a vector of the retention time of the MSN peaks stored as a variable in 
    # the xcmsRaw class
    
    # extract a composite msn spectrum where it exists for every ms1 peak
    for (i in 1:nrow(ms1peaks)) { # for each ms1 peak
      ActualParentPeakID <- 0 # variable to store the parent peak ID
      if (ms1peaks[i, "sample"] == NumXcmsPath) {
        # If the current peak (row i in the ms1peaks) is from the current sample (NumXcmsPath/
        # the current mzXML file) 
        
        # indices of all msn peaks where msn precursor mass is within the ms1 peak mz and rt range
        msn.idx <- which(precursor.mz >= npMinMz[i] & precursor.mz <= npMaxMz[i] & msn.rt >= npMinRt[i] & 
                           msn.rt <= npMaxRt[i])
        ## This is a key step! The msn scans which belong to msn1peak 'i' are identified and indexed if their
        ## precursor mz and rt fall within the min and max ranges of the mspeak1 peak.
        
        if (length(msn.idx) > 0) { # If matches are identified
          MzTable <- NULL
          # Initialise a new variable MzTable, set to NULL
          ActualParentPeakID <- i
          # Store the parent peak ID, reference to the MS1peak
          
          #### PRECURSOR INTENSITY CRITERIA ######
          
          # Single msn spectrum with highest precursor mass intensity
          if (msnSelect == "precursor_int") { 
            # if the msn selection criteria is the most intense precursor ion
            precursor.int <- xr@msnPrecursorIntensity[msn.idx]
            # A vector stores the precursor intensities of the MSN peaks from the xcmsRaw object 
            # which match the criteria of the MS1 peak
            msn.id <- msn.idx[which.max(precursor.int)]
            # Store the id of the msn peak whose precursor ion has the greatest intensity
            representative.msn.id <- msn.id
            # Unsure why the id of the precursor ion with the greatest intensity is stored 
            # into a second variable
            
            ### DETERMINE THE START AND END SCANINDEX OF THE MSN SPECTRUM #####
            
            if (msn.id < length(xr@msnScanindex)) {
              # if the msn.id is less than the total number of msnScanIndex in the xcmsRaw class
              start.id <- xr@msnScanindex[msn.id] + 1
              # the starting index is the scanIndex one greater than the scanIndex allocated to
              # the MSN peak with the greatest precursor intensity (msn.id). It is unclear why
              # the start is one greater than the ScanIndex of msn.id
              end.id <- xr@msnScanindex[msn.id + 1]
              # The end.id is determined from the ScanIndex of the peak after the MSN peak with
              # the greatest precursor intensity (msn.id+1)
            } else {
              ## As far as I can tell, this makes no sense...how could you have an MSN peak index which
              ## is greater than the number of msnScanIndices??
              start.id <- xr@msnScanindex[msn.id] + 1
              # The start.id remains the same as before
              end.id <- xr@env$msnMz
              # However, the end.id seems unclear to me...from documentation - mz = concatenated m/z
              # values for all scans
              # Having tried to print it out...this doesn't make any sense as this returns a vector of
              # all the msn peak masses.
            }
            MzTable <- new("matrix", ncol = 2, nrow = length(xr@env$msnMz[start.id:end.id]), data = c(xr@env$msnMz[start.id:end.id], 
                    xr@env$msnIntensity[start.id:end.id]))
            ## Populate the mzTable, consisting of two columns and the number of rows corresponding to the number of
            ## msnMz peaks which belong to the spectrum. Add in the intesties for each msn peak.
            colnames(MzTable) <- c("mz", "intensity")
            ## Label the columns mz and intensity
          }
          
          
          #### MS1 RETENSION TIME CRITERIA ######
          
          # Single msn spectrum with closest rt match to ms1peak rt
          if (msnSelect == "ms1_rt") {
            msn.id <- msn.idx[which.min(abs(npRt[i] - msn.rt[msn.idx]))]
            representative.msn.id <- msn.id
            # abs returns the absolute value of the retention time of the ms1peak minus the retention time
            # of the candidate msn peaks. The which.min selects the index of the lowest value. Therefore msn.id
            # is the index of the msn peak with the closest rt match to the ms1 peak.
            
            ## Determining the spectrum from the msnScanIndex is the same as above.
            if (msn.id < length(xr@msnScanindex)) {
              start.id <- xr@msnScanindex[msn.id] + 1
              end.id <- xr@msnScanindex[msn.id + 1]
            } else {
              start.id <- xr@msnScanindex[msn.id] + 1
              end.id <- xr@env$msnMz
            }
            MzTable <- new("matrix", ncol = 2, nrow = length(xr@env$msnMz[start.id:end.id]), data = c(xr@env$msnMz[start.id:end.id], 
                      xr@env$msnIntensity[start.id:end.id]))
            colnames(MzTable) <- c("mz", "intensity")
            ## Creates a table with the spectrum of peaks from the start to end id, containing the mass
            ## and intensities of the peaks
            MzTable <- MzTable[which(MzTable[, "intensity"]/max(MzTable[, "intensity"]) > min.rel.int), , 
                               drop = F]
            ## Edit the MzTable to only include msn peaks which have an intensity greater than the min.rel.int
            ## parameter which is set as a default in the initial parameters.
          }
          
          
          #### ALL ASSOCIATED PEAKS ARE TO BE USED ######
          
          # if all msn spectra associated with the ms1 peak are to be used
          if (msnSelect == "all") {
            # representative msn.id = closest RT
            representative.msn.id <- msn.idx[which.min(abs(npRt[i] - msn.rt[msn.idx]))]
            ## From the candidate list of msnPeaks, determine the absolute value of the difference between
            ## the retention time of the ms1Peak and the msnPeak
            ## Then determine the candidate with the lowest difference, which is representative.msn.id
            
            c.MzTable <- NULL
            count <- 0
            # extract msn mz and intensity values for every in-range msn scan
            for (msn.id in msn.idx) {
              count <- count + 1
              ## Determine the start and end id's using the ScanIndex as before
              if (msn.id < length(xr@msnScanindex)) {
                start.id <- xr@msnScanindex[msn.id] + 1
                end.id <- xr@msnScanindex[msn.id + 1]
              } else {
                start.id <- xr@msnScanindex[msn.id] + 1
                end.id <- xr@env$msnMz
              }
              
              c.MzTable <- rbind(c.MzTable, cbind(msn.id = rep(msn.id, length(xr@env$msnMz[start.id:end.id])), 
                      mz = xr@env$msnMz[start.id:end.id], intensity = xr@env$msnIntensity[start.id:end.id]))
              ## Append the spectrum for each matching peak to the msnTable
            }
            
            # create an intensity-weighted mz vector and an intensity 
            # matrix filled within match.ppm, with missing mz vals as NA
            weighted.mz <- numeric()
            ## creates a new numeric variable
            msn.intensity <- list()
            ## initialise a new list
            int <- c.MzTable[, "intensity"]
            ## Store the intensities of the spectrum
            count <- 0
            ## For some reason the count variable is reset to zero.
            while (any(!is.na(int))) { # while non of the intensities are missing
              ## is.na returns true for missing values
              count <- count + 1
              max.idx <- which.max(int)
              ## The max.idx corresponds to the msnPeak with the highest intensity
              mzmin <- c.MzTable[max.idx, "mz"] - (match.ppm/1e+06 * c.MzTable[max.idx, "mz"])
              ## the offset is the match.ppm/1e6*mass of the peak
              ## the min and max are determined this way
              ## mzmin and mzmax return a named number
              mzmax <- c.MzTable[max.idx, "mz"] + (match.ppm/1e+06 * c.MzTable[max.idx, "mz"])
              mz.idx <- which(c.MzTable[, "mz"] >= mzmin & c.MzTable[, "mz"] <= mzmax)
              ## the msnPicks are selected based criteria of falling within the min and max mz range
              
              weighted.mz[count] <- weighted.mean(c.MzTable[mz.idx, "mz"], c.MzTable[mz.idx, "intensity"])
              ## The weighted mean is calculated from the mass of the msnPeaks offset by the intensity of those peaks
              
              msn.intensity[[count]] <- c.MzTable[mz.idx, , drop = F][match(msn.idx, c.MzTable[mz.idx, "msn.id", 
                              drop = F]), "intensity"]
              ## The msn.intensity is the list, which accumulates a list of intensities
              
              int[mz.idx] <- NA
            }
            msn.intensity <- do.call("cbind", msn.intensity)
            ## From the list of intensities, a table is generated consisting of a single row of all intensities
            
            # calculate mean intensities, penalizing NA intensities by assigning them as 0
            msn.intensity.zeroed <- msn.intensity
            ## Create a copy of the msn.intensity table
            msn.intensity.zeroed[which(is.na(msn.intensity.zeroed))] <- 0
            ## Apply a zero value to the entries which are NA
            mean.msn.intensity <- apply(msn.intensity.zeroed, 2, mean)
            ## apply indicates that the function 'mean' should be applied to all columns (2)
            MzTable <- cbind(mz = weighted.mz, intensity = mean.msn.intensity)
            ## Add in the peaks to the MzTable
          }
          
          ###### APPLY FILTER TO REMOVE MS2 NOISE PEAKS ######
          
          
          ###### NO FILTER APPLIED #######
          
          
          if (specFilter == "none") {
            npeaks <- MzTable[order(MzTable[, "intensity"], decreasing = T), , drop = F]
          }
          
          ##### APPLY SPECPEAKS FILTER #####
          
          
          if (specFilter == "specPeaks") {
            MzTable <- MzTable[order(MzTable[, "mz"], decreasing = F), , drop = F]
            # Order the table of msnPeaks in order of mz
            npeaks <- specPeaks(MzTable, sn = sn, mzgap = mzgap)[, c("mz", "intensity"), drop = F]
            # Given a spectrum, identify and list significant peaks as determined by several criteria
            # Consolidates the list of msn peaks
          }
          
          ##### APPLY CORRELATION FILTER #####
          
          if (specFilter == "cor") {
            if (nrow(msn.intensity) > 4) {
              precursor.int <- xr@msnPrecursorIntensity[msn.idx]
              cor.mat <- cbind(precursor.int, msn.intensity)
              cor.result <- rcorr(cor.mat)
              cor.summary <- cbind(mz = weighted.mz, intensity = mean.msn.intensity, r = cor.result$r[2:nrow(cor.result$r), 
                                                                                                      1], n = cor.result$n[2:nrow(cor.result$n), 1], p = cor.result$P[2:nrow(cor.result$P), 1])
              npeaks <- cor.summary[which(cor.summary[, "r"] > min.r & cor.summary[, "n"] >= 4 & cor.summary[, 
                                                                                                             "p"] < 0.05), c("mz", "intensity"), drop = F]
            } else {
              cat("ms1 peaktable row ", i, " has only ", nrow(msn.intensity), " msn spectra; reverting to specPeaks", 
                  "\n", sep = " ")
              MzTable <- MzTable[order(MzTable[, "mz"], decreasing = F), , drop = F]
              npeaks <- specPeaks(MzTable, sn = sn, mzgap = mzgap)[, c("mz", "intensity"), drop = F]
            }
          }
          
          if (nrow(npeaks) > 0) {
            # only keep peaks with relative intensities >= min.rel.int
            npeaks <- npeaks[which(npeaks[, "intensity"]/max(npeaks[, "intensity"]) >= min.rel.int), , drop = F]
            # only keep peaks whose masses are smaller than precursor mass - min.diff
            npeaks <- npeaks[which(npeaks[, "mz"] < (npMz[i] - min.diff)), , drop = F]
            if (nrow(npeaks) > 0) {
              # only keep a maximum of max.frags peaks
              npeaks <- npeaks[1:min(max.frags, nrow(npeaks)), , drop = F]
              for (numPeaks in 1:nrow(npeaks)) {
                # for every picked msn peak
                PeakNr <- PeakNr + 1
                # increasing peakid
                npPeakID[PeakNr] <- PeakNr
                npMSnParentPeakID[PeakNr] <- ActualParentPeakID
                npMsLevel[PeakNr] <- xr@msnLevel[representative.msn.id]
                npRt[PeakNr] <- xr@msnRt[representative.msn.id]
                npMz[PeakNr] <- npeaks[numPeaks, "mz"]
                npIntensity[PeakNr] <- npeaks[numPeaks, "intensity"]
                npSample[PeakNr] <- NumXcmsPath
                npCollisionEnergy[PeakNr] <- xr@msnCollisionEnergy[representative.msn.id]
              }
            }
          }
        }
      }
    }
    if (ActualParentPeakID == 0) {
      numAloneSpecs <- numAloneSpecs + 1
    }
  }
  
  ###### STORE OUTPUT IN XCMSFRAGMENT OBJECT #######
  
  fragmentColnames <- c("peakID", "MSnParentPeakID", "msLevel", "rt", "mz", "intensity", "Sample", "GroupPeakMSn", 
                        "CollisionEnergy")
  npGroupPeakMSn <- rep(0, length(npSample))
  # add group information if it exists
  gv <- groupval(xs)
  if (length(gv) > 0) {
    for (i in 1:nrow(gv)) {
      npGroupPeakMSn[npPeakID %in% gv[i, ]] <- i
      npGroupPeakMSn[npMSnParentPeakID %in% gv[i, ]] <- i
    }
  }
  
  object@peaks <- new("matrix", nrow = length(npMz), ncol = length(fragmentColnames), data = c(npPeakID, npMSnParentPeakID, 
                    npMsLevel, npRt, npMz, npIntensity, npSample, npGroupPeakMSn, npCollisionEnergy))
  ## Object parameter peaks is used to store a new matrix with the msn data
  colnames(object@peaks) <- fragmentColnames
  
  
  cat(length(npPeakID), "Peaks picked,", numAloneSpecs, "MSn-Specs ignored.\n")
  object
}
