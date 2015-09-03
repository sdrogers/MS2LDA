### This is the initial peak detection workflow we have before --- using Tony's script ###
### but modified to allow us to specify which MS1 full scan data to use ###
run_create_peak_method_2 <- function(config) {
 
    full_scan_file <- config$input_files$full_scan_file
    fragmentation_file <- config$input_files$fragmentation_file_m2
    
    print("Running create_peak_method #2")
    
    ## do peak detection on full scan file
    xset_full <- xcmsSet(files=full_scan_file, method=config$MS1_XCMS_peakpicking_settings$method, ppm=config$MS1_XCMS_peakpicking_settings$ppm, snthresh=config$MS1_XCMS_peakpicking_settings$snthres, peakwidth=c(config$MS1_XCMS_peakpicking_settings$peakwidth_from,config$MS1_XCMS_peakpicking_settings$peakwidth_to),
                           prefilter=c(config$MS1_XCMS_peakpicking_settings$prefilter_from,config$MS1_XCMS_peakpicking_settings$prefilter_to), mzdiff=config$MS1_XCMS_peakpicking_settings$mzdiff, integrate=config$MS1_XCMS_peakpicking_settings$integrate, fitgauss=config$MS1_XCMS_peakpicking_settings$fitgauss, verbose.column=config$MS1_XCMS_peakpicking_settings$verbose.column)
    xset_full <- group(xset_full)
    
    # do peak detection on fragmentation file
    xset <- xcmsSet(files=fragmentation_file, method=config$MS1_XCMS_peakpicking_settings$method, ppm=config$MS1_XCMS_peakpicking_settings$ppm, snthresh=config$MS1_XCMS_peakpicking_settings$snthres, peakwidth=c(config$MS1_XCMS_peakpicking_settings$peakwidth_from,config$MS1_XCMS_peakpicking_settings$peakwidth_to),
                    prefilter=c(config$MS1_XCMS_peakpicking_settings$prefilter_from,config$MS1_XCMS_peakpicking_settings$prefilter_to), mzdiff=config$MS1_XCMS_peakpicking_settings$mzdiff, integrate=config$MS1_XCMS_peakpicking_settings$integrate, fitgauss=config$MS1_XCMS_peakpicking_settings$fitgauss, verbose.column=config$MS1_XCMS_peakpicking_settings$verbose.column)
    xset <- group(xset)
    
    # run modified Tony's script
    source('xcmsSetFragments.modified.R')
    frags <- xcmsSetFragments(xset, xset_full,
                              cdf.corrected=config$ms1_ms2_pairing_parameters$cdf.corrected, 
                              min.rel.int=config$filtering_parameters_MassSpectrometry_related$min.rel.int, 
                              max.frags=config$filtering_parameters_MassSpectrometry_related$mass.frags, 
                              msnSelect=c(config$ms1_ms2_pairing_parameters$msnSelect), 
                              specFilter=c(config$ms1_ms2_pairing_parameters$specFilter), 
                              match.ppm=config$ms1_ms2_pairing_parameters$match.ppm, 
                              sn=config$filtering_parameters_MassSpectrometry_related$sn, 
                              mzgap=config$filtering_parameters_MassSpectrometry_related$mz_gap, min.r=config$ms1_ms2_pairing_parameters$min.r, 
                              min.diff=config$ms1_ms2_pairing_parameters$min.diff)
    peaks <- as.data.frame(frags@peaks)
    
    return(peaks)
    
}