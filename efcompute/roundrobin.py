import sys
import fractions
from math import ceil,floor
from ef_assigner import ef_assigner
# Set infinity
infinite = sys.maxint


atom_masses = {'C':12.00000000000,
                'H':1.00782503214,
                'N':14.00307400524,
                'O':15.99491462210,
                'P':30.97376151200,
                'S':31.97207069000}

def find_n(n, p, d):
    minimum = infinite
    for q, elem in enumerate(n):
        if q % d == p:
            if elem < minimum:
                minimum = elem
    return minimum


def round_robin(a):
    
    k = len(a)
    a1 = a[0]
    n = [infinite for r in range(0, a1)]
    n[0] = 0
    

    rr = []
    rr.append(list(n))
    

    for i in range(1, k):
        d = fractions.gcd(a1, a[i])
        for p in range(0, d):
            new_n = find_n(n, p, d)
            if new_n < infinite:
                for repeat in range(1, a1/d):
                    new_n = new_n + a[i]
                    r = new_n % a1
                    new_n = min(new_n, n[r])
                    n[r] = new_n
        
        rr.append(list(n))

    return rr
    


def find_all(mass,i,c,a,rt):
    if i == 0:
        c[0] = mass/a[0]
        formulas.append(list(c))
        return
    else:
        lcm = a[0]*a[i] / fractions.gcd(a[0],a[i])
        l = lcm/a[i]
        for j in range(0,l):
            c[i] = j
            m = mass - j*a[i]
            r = m % a[0]
            lbound = rt[i-1][r]
            while m >= lbound:
                find_all(m,i-1,c,a,rt)
                m = m - lcm
                c[i] = c[i] + l


def get_dictionary(atoms,scale_factor):
    atom_dict = []
    for a in atoms:
        atom_dict.append(int(ceil(atom_masses[a]*scale_factor)))
    return atom_dict

def make_formula_string(atoms,counts):
    f_string = ""
    for i,a in enumerate(atoms):
        if counts[i] == 0:
            pass
        elif counts[i] == 1:
            f_string += a
        else:
            f_string += "{}{}".format(a,counts[i])
    return f_string

def get_formula_mass(atoms,counts):
    mass = 0.0
    for i,a in enumerate(atoms):
        mass += atom_masses[a]*counts[i]
    return mass

def find_formulas(precursor_mass_list,ppm,atoms = ['C','H','N','O','P','S'],scale_factor=1000):
    global formulas

    a = get_dictionary(atoms,scale_factor)
    print "Computing round robin table"
    rr = round_robin(a)
    formulas_out = {}

    # Compute correction factor for upper bound
    delta = 0
    for i in atoms:
        delta_i = (ceil(scale_factor*atom_masses[i]) - scale_factor*atom_masses[i])/atom_masses[i]
        if delta_i > delta:
            delta = delta_i

    print "Finding formulas at {}ppm".format(ppm)
    for precursor_mass in precursor_mass_list:

        k = len(rr)
        c = [0 for i in range(0, k)]

        print "Searching for {}".format(precursor_mass)
        formulas = []

        ppm_error = ppm*precursor_mass/1e6
        lower_bound = precursor_mass - ppm_error
        upper_bound = precursor_mass + ppm_error

        int_lower_bound = int(ceil(lower_bound*scale_factor))
        int_upper_bound = int(floor(scale_factor*upper_bound + delta*upper_bound))

        for int_mass in range(int_lower_bound,int_upper_bound+1):
            find_all(int_mass,k-1,c,a,rr)

        print "\t found {}".format(len(formulas))

        formulas_out[precursor_mass] = list(formulas)

    return formulas_out


if __name__=='__main__':

    atoms = ['C','H','N','O','P','S']
    test_molecules = [[0,2,0,1,0,0],[1,0,0,2,0,0],[0,0,0,2,0,0],[8,10,4,2,0,0]]
    precursor_mass_list = []
    for test_molecule in test_molecules:
        precursor_mass_list.append(get_formula_mass(atoms,test_molecule))
    

    # formulas_out = find_formulas(precursor_mass_list,1)
    ef = ef_assigner()
    formulas_out = ef.find_formulas(precursor_mass_list,1)

    for p in precursor_mass_list:
        print "Mass: {}".format(p)
        for f in formulas_out[p]:
            print "\t{} ({})".format(make_formula_string(atoms,f),get_formula_mass(atoms,f))
