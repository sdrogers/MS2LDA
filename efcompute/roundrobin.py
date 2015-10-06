
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

    atoms = ['C','H','N','O','P','S']
    test_molecules = [[0,2,0,1,0,0],[1,0,0,2,0,0],[0,0,0,2,0,0],[8,10,4,2,0,0],[1,1,1,1,1,1]]
    precursor_mass_list = []
    for test_molecule in test_molecules:
        precursor_mass_list.append(get_formula_mass(atoms,test_molecule))
    

    # formulas_out = find_formulas(precursor_mass_list,1)
    ef = ef_assigner()
    formulas_out = ef.find_formulas(precursor_mass_list,1)

    for p in precursor_mass_list:
        print "Mass: {}".format(p)
        for f in formulas_out[p]:
            s,m = make_formula_string(f)
            print "\t{} ({})".format(s,m)

    from golden_rules import golden_rules

    g = golden_rules()
    filtered_out = {}
    reasons = {}
    for p in precursor_mass_list:
        filtered_out[p],reasons[p] = g.filter_list(formulas_out[p])

    print
    print
    print "FILTERED"
    print
    print
    for p in precursor_mass_list:
        print "Mass: {}".format(p)
        for f in filtered_out[p]:
            s,m = make_formula_string(f)
            print "\t{} ({})".format(s,m)

    print
    print
    print "REASONS"
    print
    print

    for p in precursor_mass_list:
        print "Mass: {}".format(p)
        for f in reasons[p]:
            print "\t{} {}".format(f,reasons[p][f])

