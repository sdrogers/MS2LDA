import pandas as pd
import subprocess
import os
import json
import pprint
import tempfile
import shutil

ms1_filename = '../../input/final/Beer_3_full1_5_2E5_pos_ms1.csv'
ms2_filename = '../../input/final/Beer_3_full1_5_2E5_pos_ms2.csv'
ms1 = pd.read_csv(ms1_filename, index_col=0)
ms2 = pd.read_csv(ms2_filename, index_col=0)
counter = 0
pp = pprint.PrettyPrinter(depth=4)
lim = 10
for row_index, row in ms1.iterrows():

    parent_peak_id = int(row.peakID)
    parent_mass = row.mz
    parent_intensity = row.intensity
    print "Parent peakID=%s, mass=%s, intensity=%s" % (parent_peak_id, parent_mass, parent_intensity)
    children = ms2[ms2.MSnParentPeakID==parent_peak_id]
    fragment_mzs = children.mz.values
    fragment_intensities = children.intensity.values
    n_frags = len(fragment_mzs)
    print "Fragment masses="
    print fragment_mzs
    print "Fragment intensities="
    print fragment_intensities
    print
    
    # create temp mgf file
    print "mgf file generated="
    mgf = "BEGIN IONS\n"
    mgf += "PEPMASS=" + str(parent_mass) + "\n"
    mgf += "MSLEVEL=1\n"
    mgf += "CHARGE=1+\n"
    mgf += str(parent_mass) + " " + str(parent_intensity) + "\n"
    mgf += "END IONS\n"
    mgf += "\n"
    mgf += "BEGIN IONS\n"
    mgf += "PEPMASS=" + str(parent_mass) + "\n"
    mgf += "MSLEVEL=2\n"
    mgf += "CHARGE=1+\n"
    for n in range(n_frags):
        mgf += str(fragment_mzs[n]) + " " + str(fragment_intensities[n]) + "\n"
    mgf += "END IONS"
    print mgf

    temp_dir = tempfile.mkdtemp()
    with open("temp.mgf", "w") as text_file:
        text_file.write(mgf)
    args = ['linux64/sirius', '-p', 'orbitrap', '-O', 'json', '-o', temp_dir, 'temp.mgf']
    subprocess.call(args)

    # read the first file    
    files = sorted(os.listdir(temp_dir))
    first_filename = os.path.join(temp_dir, files[0])
    json_data = open(first_filename).read()
    data = json.loads(json_data)
    print "JSON OUTPUT"
    pp.pprint(data)
    
    # delete all the temp files
    shutil.rmtree(temp_dir)
        
    print "=============================================================="    
    counter += 1
    if counter >= lim:
        break