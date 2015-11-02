import fractions
from math import ceil,floor
from golden_rules import golden_rules
from ef_constants import INFINITE, ATOM_NAME_LIST, ATOM_MASSES, PROTON_MASS, DEFAULT_RULES_SWITCH

class ef_assigner(object):
    
    def __init__(self, scale_factor=1000, enforce_ppm=True, do_7_rules=True, 
                 second_stage=False, rule_8_max_occurrences=None,
                 verbose = True):

        self.verbose = verbose
        self.atoms = list(ATOM_NAME_LIST) # copy
        self.atom_masses = dict(ATOM_MASSES)

        self.do_7_rules = do_7_rules
        if self.do_7_rules:
            
            # if the limit on max occurrences of atoms is provided, then
            # turn on the filtering on the occurrences (so-called rule #8)
            rule_switch = list(DEFAULT_RULES_SWITCH)
            if rule_8_max_occurrences is None:
                rule_switch[7] = False
            else:
                rule_switch[7] = True                

            # if second stage clustering, then also include C13, F and Cl
            # otherwise remove them from the list of atoms to be considered
            if not second_stage:
                self.atoms.remove('C13')
                self.atoms.remove('F')
                self.atoms.remove('Cl')
                del self.atom_masses['C13']
                del self.atom_masses['F']
                del self.atom_masses['Cl']

            self.gr = golden_rules(rule_switch, rule_8_max_occurrences)
        
        self.scale_factor = scale_factor
        self.a = self._get_dictionary()
        self.rr = self._round_robin()
        self.enforce_ppm = enforce_ppm
        
        # Compute correction factor for upper bound
        self.delta = 0
        for i in self.atoms:
            delta_i = (ceil(scale_factor*self.atom_masses[i]) - scale_factor*self.atom_masses[i])/self.atom_masses[i]
            if delta_i > self.delta:
                self.delta = delta_i
        # print self.delta

    def find_formulas(self, mass_list, ppm=5, polarisation="None", max_mass_to_check=INFINITE):
        
        polarisation = polarisation.lower()
        
        # used to accumulate values during the recursive calls in _find_all
        # TODO: better if we can get rid of the global keyword
        global formulas

        # check for conditional mass tolerance
        if type(ppm) is list:
            cond_mass_tol = True
        else:
            cond_mass_tol = False
        
        formulas_out = {}
        top_hit_string = []
        n = 0
        total = len(mass_list)
        precursor_mass_list = []
        for mass in mass_list:
            
            n += 1
            k = len(self.rr)
            c = [0 for i in range(0, k)]

            formulas = []

            # compute the right precursor mass, given the polarisation
            if polarisation == "pos":
                precursor_mass = mass - PROTON_MASS
            elif polarisation == "neg":
                precursor_mass = mass + PROTON_MASS
            else:
                precursor_mass = mass
            precursor_mass_list.append(precursor_mass)

            # get the mass tolerance
            if cond_mass_tol:
                conditional_ppm = self._get_conditional_ppm(precursor_mass, ppm)
            else:
                conditional_ppm = ppm # unchanged, this should be a float

            # always return None for all precursor masses above max_ms1
            if precursor_mass > max_mass_to_check:
                top_hit_string.append(None)
                continue
            elif self.verbose:
                    print "Searching for neutral mass %f (%d/%d) at tolerance %d ppm" % (precursor_mass, n, total, 
                                                                                         conditional_ppm)

            # find all the candidate formulae                
            ppm_error = conditional_ppm*precursor_mass/1e6
            lower_bound = precursor_mass - ppm_error
            upper_bound = precursor_mass + ppm_error
            int_lower_bound = int(ceil(lower_bound*self.scale_factor))
            int_upper_bound = int(floor(upper_bound*self.scale_factor + self.delta*upper_bound))
            for int_mass in range(int_lower_bound, int_upper_bound+1):
                self._find_all(int_mass, k-1, c, conditional_ppm, precursor_mass)

            formulas_out[precursor_mass] = []
            for f in formulas:
                formula = {}
                for i,a in enumerate(self.atoms):
                    formula[a] = f[i]
                formulas_out[precursor_mass].append(formula)

            # check with 7 golden rules
            if self.do_7_rules:
                filtered_formulas_out = {}
                filtered_formulas_out[precursor_mass], passed, failed = self.gr.filter_list(formulas_out[precursor_mass])
                formulas_out[precursor_mass] = filtered_formulas_out[precursor_mass]
            
            # adjust the amount of hydrogen to print the charged formula -- based on polarity
            if polarisation == "pos":
                for f in formulas_out[precursor_mass]:
                    f['H'] += 1
            elif polarisation == "neg":
                for f in formulas_out[precursor_mass]:
                    if f['H'] >= 1:
                        f['H'] -= 1

            # If there is no top hit string, then just set None to the resulting list
            if len(formulas_out[precursor_mass]) == 0:
                top_hit_string.append(None)
                continue
            else:
                
                # else find the formula closest in mass to the theoretical mass 
                closest = None
                for f in formulas_out[precursor_mass]:
                    mass = 0.0
                    f_string = ""
                    for atom in f:
                        
                        mass += f[atom]*self.atom_masses[atom]

                        # print C13 as [C13]
                        atom_str = atom
                        if atom_str == 'C13':
                            atom_str = '[C13]'
                        if f[atom]>1:
                            f_string += "{}{}".format(atom_str, f[atom])
                        elif f[atom] == 1:
                            f_string += "{}".format(atom_str)
                    
                    er = abs(mass - precursor_mass)
                    # print er, f_string
                    if closest == None:
                        best_er = er
                        closest = f_string
                    elif best_er > er:
                        best_er = er
                        closest = f_string                
                top_hit_string.append(closest)

                if self.verbose:
                    if len(formulas) > 0:
                        if len(formulas_out[precursor_mass]) > 0:
                            print "- found " + str(len(formulas)) + " candidate(s), best formula = " + closest
                        else:
                            print "- found " + str(len(formulas)) + " candidate(s), nothing after filtering"                        
                    else:
                        print "- no candidate formula found"

        return formulas_out, top_hit_string, precursor_mass_list
    
    def _get_conditional_ppm(self, mass, ppm_list):
        ''' Get the conditional annotation ppm for the specified mass '''

        # assume ppm_list always contains 2 entries
        first_item = ppm_list[0]
        second_item = ppm_list[1]
        lower_bound = first_item[0]
        upper_bound = second_item[0]

        # find the conditional tolerance to use for this mass
        if mass > lower_bound and mass < upper_bound:
            higher_tol = second_item[1]
            return higher_tol
        elif mass <= lower_bound:
            lower_tol = first_item[1]
            return lower_tol
        elif mass >= upper_bound:
            # use the higher tolerance anyway
            higher_tol = second_item[1]
            return higher_tol
    
    def _find_all(self, mass, i, c, ppm, precursor_mass):

        if i == 0:
            c[0] = mass/self.a[0]
            # The following corrects for the fact that at low scale factors we
            # will find things at much more than the specificed ppm
            if self.enforce_ppm:
                molecule_mass = 0.0
                for i,atom in enumerate(self.atoms):
                    molecule_mass += self.atom_masses[atom]*c[i]
                if abs(molecule_mass - precursor_mass)/precursor_mass <= 1e-6*ppm:
                    formulas.append(list(c))

                return
            else:
                formulas.append(list(c))
            return
        else:
            lcm = self.a[0]*self.a[i] / fractions.gcd(self.a[0],self.a[i])
            l = lcm/self.a[i]
            for j in range(0,l):
                c[i] = j
                m = mass - j*self.a[i]
                r = m % self.a[0]
                lbound = self.rr[i-1][r]
                while m >= lbound:
                    self._find_all(m, i-1, c, ppm, precursor_mass)
                    m = m - lcm
                    c[i] = c[i] + l    

    def _get_dictionary(self):
        
        atom_dict = []
        for a in self.atoms:
            atom_dict.append(int(ceil(self.atom_masses[a]*self.scale_factor)))
        return atom_dict

    def _round_robin(self):
    
        k = len(self.a)
        a1 = self.a[0]
        n = [INFINITE for r in range(0, a1)]
        n[0] = 0
        
        rr = []
        rr.append(list(n))

        for i in range(1, k):
            d = fractions.gcd(a1, self.a[i])
            for p in range(0, d):
                new_n = self._find_n(n, p, d)
                if new_n < INFINITE:
                    for repeat in range(1, a1/d):
                        new_n = new_n + self.a[i]
                        r = new_n % a1
                        new_n = min(new_n, n[r])
                        n[r] = new_n
            
            rr.append(list(n))

        return rr
    
    def _find_n(self, n, p, d):

        minimum = INFINITE
        for q, elem in enumerate(n):
            if q % d == p:
                if elem < minimum:
                    minimum = elem
        return minimum
