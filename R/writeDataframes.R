write_output <- function(ms1, ms2, fragment_df, neutral_loss_df, config) {
    
    # construct the output filenames
    prefix <- config$input_files$prefix
    fragments_out <- paste(c(prefix, '_fragments.csv'), collapse="")
    losses_out <- paste(c(prefix, '_losses.csv'), collapse="")
    # mzdiffs_out <- paste(c(prefix, '_mzdiffs.csv'), collapse="")
    ms1_out <- paste(c(prefix, '_ms1.csv'), collapse="")
    ms2_out <- paste(c(prefix, '_ms2.csv'), collapse="")
    
    # write stuff out
    write.table(ms1, file=ms1_out, col.names=NA, row.names=T, sep=",")
    write.table(ms2, file=ms2_out, col.names=NA, row.names=T, sep=",")    
    write.table(fragment_df, file=fragments_out, col.names=NA, row.names=T, sep=",")
    write.table(neutral_loss_df, file=losses_out, col.names=NA, row.names=T, sep=",")
    # write.table(mz_diff_df, file=mzdiffs_out, col.names=NA, row.names=T, sep=",")    
    
}