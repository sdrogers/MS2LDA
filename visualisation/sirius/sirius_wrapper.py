import pandas as pd
import subprocess
import os
import json
import tempfile
import shutil
import pprint
import platform

def annotate_sirius(ms1, ms2, sirius_platform='orbitrap', mode="pos", verbose=False):

    if mode != "pos" and mode != "neg":
        raise ValueError("mode is either 'pos' or 'neg'")

    ms1 = ms1.copy()
    ms2 = ms2.copy()

    total_ms1 = 0
    total_ms2 = 0
    devnull = open(os.devnull, 'w')
    for ms1_row_index, ms1_row in ms1.iterrows():

        # make mgf data    
        parent_peak_id = int(ms1_row.peakID)
        children = ms2[ms2.MSnParentPeakID==parent_peak_id]
        mgf = make_mgf(ms1_row, children, mode)
        if verbose:
            print mgf
    
        # write temp mgf file
        temp_dir = tempfile.mkdtemp()
        temp_filename = tempfile.mktemp()
        with open(temp_filename, "w") as text_file:
            text_file.write(mgf)

        # run sirius on the temp mgf file
        starting_dir = os.getcwd()
        try:

            # detect OS and pick the right executable for SIRIUS
            system = platform.system()
            arch = platform.architecture()[0]
            
            valid_platform = False
            if system == 'Linux':
                if arch == '64bit':
                    valid_platform = True
                    sirius_dir = 'linux64'
                    sirius_exec = 'sirius'                    
            elif system == 'Windows':
                if arch == '32bit':
                    valid_platform = True
                    sirius_dir = 'win32'
                    sirius_exec = 'sirius.exe'                    
                elif arch == '64bit':
                    valid_platform = True
                    sirius_dir = 'win64'
                    sirius_exec = 'sirius.exe'                    

            if not valid_platform:
                raise ValueError(system + " " + arch + " is not supported")

            current_script_dir = os.path.dirname(os.path.realpath(__file__))
            full_exec_dir = os.path.join(current_script_dir, sirius_dir)
            full_exec_path = os.path.join(full_exec_dir, sirius_exec)
            os.chdir(full_exec_dir)
            args = [full_exec_path, '-p', sirius_platform, '-s', 'omit', '-O', 'json', '-o', temp_dir, temp_filename]                
            if verbose:
                subprocess.check_call(args)
            else:
                subprocess.check_call(args, stdout=devnull, stderr=devnull)
        
            # read the first file produced by sirius    
            files = sorted(os.listdir(temp_dir))
            if len(files) == 0: # sometimes nothing is produced?
                continue
            
            first_filename = os.path.join(temp_dir, files[0])
            json_data = open(first_filename).read()
            data = json.loads(json_data)    
        
        except subprocess.CalledProcessError, e:
            print "Sirius produced error: " + str(e)
            break # stop the loop
        finally:
            # delete all the temp files regardless
            shutil.rmtree(temp_dir)
            os.remove(temp_filename)
            # restore current directory
            os.chdir(starting_dir)
            
        # put the results back into the ms1 and ms2 df
        overall_score = data['annotations']['score']['total']
        if overall_score > 0.01: # ignore 0 score
    
            if verbose:
                print "JSON OUTPUT"
                pp = pprint.PrettyPrinter(depth=4)
                pp.pprint(data)
                
            # annotate the children ms2 rows
            annot_count = 0
            for child_row_index, child_row in children.iterrows():
                
                # get the mz of ms2 peak
                child_mz = child_row.mz
    
                # loop over all annotations and find matching entry
                fragment_annots = data['fragments']
                for fa in fragment_annots:
    
                    annot_mz = fa['mz']
                    close = abs(child_mz - annot_mz) < 1e-8                
                    if close:
                        ms2.loc[child_row_index, 'annotation'] = fa['molecularFormula']
                        annot_count += 1
                        break

            if annot_count > 0:            

                total_ms2 += annot_count
                print " - %s fragments annotated" % annot_count
                
                # annotate the ms1 row too
                root_formula = data['molecularFormula']
                ms1.loc[ms1_row_index, 'annotation'] = root_formula
                total_ms1 += 1
                                
    nrow_ms1 = ms1.shape[0]
    nrow_ms2 = ms2.shape[0]
    print
    print "Total annotations MS1=%s/%s, MS2=%s/%s" % (total_ms1, nrow_ms1, total_ms2, nrow_ms2)
    return ms1, ms2
            
def make_mgf(ms1_row, children, mode):

    parent_peak_id = int(ms1_row.peakID)
    parent_mass = ms1_row.mz
    parent_intensity = ms1_row.intensity

    # get the fragment info
    fragment_mzs = children.mz.values
    fragment_intensities = children.intensity.values
    n_frags = len(fragment_mzs)    
    print "Annotating parent peakID=%s, mass=%s, intensity=%s, no. of fragments=%s" % (parent_peak_id, parent_mass, 
                                                                                       parent_intensity, n_frags)
    
    # create temp mgf file
    mgf = "BEGIN IONS\n"
    mgf += "PEPMASS=" + str(parent_mass) + "\n"
    mgf += "MSLEVEL=1\n"
    if mode == "pos":
        mgf += "CHARGE=1+\n"
    elif mode == "neg":
        mgf += "CHARGE=1-\n"        
    mgf += str(parent_mass) + " " + str(parent_intensity) + "\n"
    mgf += "END IONS\n"
    mgf += "\n"
    mgf += "BEGIN IONS\n"
    mgf += "PEPMASS=" + str(parent_mass) + "\n"
    mgf += "MSLEVEL=2\n"
    if mode == "pos":
        mgf += "CHARGE=1+\n"
    elif mode == "neg":
        mgf += "CHARGE=1-\n"        
    for n in range(n_frags):
        mgf += str(fragment_mzs[n]) + " " + str(fragment_intensities[n]) + "\n"
    mgf += "END IONS"

    return mgf
            
def main():

    ms1_filename = '../../input/final/Beer_3_full1_5_2E5_pos_ms1.csv'
    ms2_filename = '../../input/final/Beer_3_full1_5_2E5_pos_ms2.csv'
    ms1 = pd.read_csv(ms1_filename, index_col=0)
    ms2 = pd.read_csv(ms2_filename, index_col=0)
    sirius_exec = '/home/joewandy/linux64/sirius'
    annot_ms1, annot_ms2 = annotate_sirius(ms1, ms2, sirius_exec)

    ms1_filename = '../../input/final/Beer_3_full1_5_2E5_pos_ms1_annotated.csv'
    ms2_filename = '../../input/final/Beer_3_full1_5_2E5_pos_ms2_annotated.csv'
    annot_ms1.to_csv(ms1_filename)
    annot_ms2.to_csv(ms2_filename)
    print "Annotated MS1 results written to " + ms1_filename
    print "Annotated MS2 results written to " + ms2_filename

if __name__ == "__main__":
    main()