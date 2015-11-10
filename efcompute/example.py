from ef_assigner import ef_assigner
from ef_constants import ATOM_MASSES, PROTON_MASS, ATOM_NAME_LIST

def get_formula_mass(atoms,counts):
    
    mass = 0.0
    for i,a in enumerate(atoms):
        mass += ATOM_MASSES[a]*counts[i]

    return mass

def make_formula_string(formula):
    
    f_string = ""
    f_mass = 0.0
    for a in formula:
        if formula[a] == 1:
            f_string += a
        elif formula[a] > 1:
            f_string += "{}{}".format(a,formula[a])
        f_mass += ATOM_MASSES[a]*formula[a]
    
    return f_string,f_mass

if __name__=='__main__':

    ATOM_NAME_LIST = ['H', 'C', 'C13', 'N', 'O', 'F', 'P', 'S', 'Cl']

    # Make some molecules and and compute their masses
    atoms = ATOM_NAME_LIST
    test_molecules = [[2,  0,  0,  0,  1,  0,  0,  0,  0], # H2O
                      [0,  1,  0,  0,  2,  0,  0,  0,  0], # CO2
                      [0,  0,  0,  0,  2,  0,  0,  0,  0], # O2
                      [10, 8,  0,  4,  0,  0,  2,  0,  0], # C8H10N4P2
                      [1,  1,  0,  1,  1,  0,  1,  0,  0], # CHNOPS
                      [7,  4,  0,  1,  0,  0,  0,  0,  0], # C4H7N
                      [0,  0,  1,  0,  2,  0,  0,  0,  0], # [C13]O2
                      [1,  0,  0,  0,  0,  1,  0,  0,  0], # HF
                      [1,  0,  0,  0,  0,  0,  0,  0,  1], # HCl
                      ]
    mass_list = []
    for test_molecule in test_molecules:
        mass_list.append(get_formula_mass(atoms,test_molecule))
    
    # Create the ef_assigner object
    ef = ef_assigner(scale_factor=100, do_7_rules=True, 
                     second_stage=False)

    polarisation = "POS"    
    for n in range(len(mass_list)):
        if polarisation == "POS":
            mass_list[n] = mass_list[n] + PROTON_MASS
        elif polarisation == "NEG":
            mass_list[n] = mass_list[n] + PROTON_MASS

    # Find the formulas for the list of masses
    formulas_out, top_hit_string, precursor_mass_list = ef.find_formulas(mass_list, ppm=10, polarisation="POS")

    # Print the output
    for precursor_mass, measured_mass in zip(precursor_mass_list, mass_list):
        print "Mass: {}".format(precursor_mass)
        for f in formulas_out[precursor_mass]:
            s, m = make_formula_string(f)
            ppm_error = 1e6*(m - float(precursor_mass))/m
            print "\t{} ({}) ({})".format(s, m, ppm_error)

    print
    print "top_hit_string: "
    for (m, s) in zip(mass_list, top_hit_string):
        print (m, s)

#     # Import the seven golden rules code (Needs more testing)
#     from golden_rules import golden_rules
# 
#     # Create a golden rules object
#     g = golden_rules()
#     filtered_out = {}
#     passed = {}
#     failed = {}
#     
#     # Loop through the masses, and filter the hits
#     for p in precursor_mass_list:
#         filtered_out[p],passed[p],failed[p] = g.filter_list(formulas_out[p])
#     
#     # Print the filtered list
#     print
#     print
#     print "FILTERED"
#     print
#     print
#     for p in precursor_mass_list:
#         print "Mass: {}".format(p)
#         for f in filtered_out[p]:
#             s,m = make_formula_string(f)
#             ppm_error = 1e6*abs(m - p)/p
#             print "\t{} ({}) (error = {})".format(s,m,ppm_error)
#     
#     # print the formulas that fail the test
#     print
#     print
#     print "FAILED"
#     print
#     print
#     for p in precursor_mass_list:
#         print "Mass: {}".format(p)
#         for f in failed[p]:
#             print "\t{} {}".format(f,failed[p][f])