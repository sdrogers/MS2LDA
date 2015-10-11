
from ef_assigner import ef_assigner


atom_masses = {'C':12.00000000000,
                'H':1.00782503214,
                'N':14.00307400524,
                'O':15.99491462210,
                'P':30.97376151200,
                'S':31.97207069000}

def get_formula_mass(atoms,counts):
    mass = 0.0
    for i,a in enumerate(atoms):
        mass += atom_masses[a]*counts[i]
    return mass

def make_formula_string(formula):
    f_string = ""
    f_mass = 0.0
    for a in formula:
        if formula[a] == 1:
            f_string += a
        elif formula[a] > 1:
            f_string += "{}{}".format(a,formula[a])
        f_mass += atom_masses[a]*formula[a]

    return f_string,f_mass

if __name__=='__main__':


    # Make some molecules and and compute their masses
    atoms = ['C','H','N','O','P','S']
    test_molecules = [[0,2,0,1,0,0],[1,0,0,2,0,0],[0,0,0,2,0,0],[8,10,4,2,0,0],[1,1,1,1,1,1]]
    precursor_mass_list = []
    for test_molecule in test_molecules:
        precursor_mass_list.append(str(get_formula_mass(atoms,test_molecule)+1.00727645199076))
    
    # Create the ef_assigner object
    ef = ef_assigner(scale_factor=1000)

    # precursor_mass_list = [130.06528 - 1.00727645199076]

    # Find the formulas for the list of masses
    formulas_out,top_hit_string = ef.find_formulas(precursor_mass_list,ppm=10,polarisation ="POS")


    # Print the output
    for p in precursor_mass_list:
        print "Mass: {}".format(p)
        for f in formulas_out[p]:
            s,m = make_formula_string(f)
            ppm_error = 1e6*(m - float(p))/m
            print "\t{} ({}) ({})".format(s,m,ppm_error)


    for s in top_hit_string:
        print s

    # # Import the seven golden rules code (Needs more testing)
    # from golden_rules import golden_rules

    # # Create a golden rules object
    # g = golden_rules()
    # filtered_out = {}
    # passed = {}
    # failed = {}

    # # Loop through the masses, and filter the hits
    # for p in precursor_mass_list:
    #     filtered_out[p],passed[p],failed[p] = g.filter_list(formulas_out[p])


    # # Print the filtered list
    # print
    # print
    # print "FILTERED"
    # print
    # print
    # for p in precursor_mass_list:
    #     print "Mass: {}".format(p)
    #     for f in filtered_out[p]:
    #         s,m = make_formula_string(f)
    #         ppm_error = 1e6*abs(m - p)/p
    #         print "\t{} ({}) (error = {})".format(s,m,ppm_error)


    # # print the formulas that fail the test
    # print
    # print
    # print "FAILED"
    # print
    # print
    # for p in precursor_mass_list:
    #     print "Mass: {}".format(p)
    #     for f in failed[p]:
    #         print "\t{} {}".format(f,failed[p][f])
