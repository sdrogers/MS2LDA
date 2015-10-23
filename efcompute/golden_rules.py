from ef_constants import ATOM_MASSES, ATOM_VALENCES, DEFAULT_RULES_SWITCH

class golden_rules(object):

    def __init__(self, rule_switch=DEFAULT_RULES_SWITCH, rule_8_max_occurrences=None):
        self.rule_switch = rule_switch
        self.rule_8_max_occurrences = rule_8_max_occurrences

    def get_formula_mass(self, formula):
        mass = 0.0
        for a in formula:
            mass += ATOM_MASSES[a]*formula[a]
        return mass

    def element_numbers_restriction(self, formula, c, h, n, o, p, s, f, cl, br):
        """
        :param formula: the formula to check if in restrictions
        :param c: maximum number of carbon
        :param h: maximum number of hydrogen
        :param n: maximum number of nitrogen
        :param o: maximum number of oxygen
        :param p: maximum number of phosphorus
        :param s: maximum number of sulfur
        :param f: maximum number of fluorine
        :param cl: maximum number of chlorine
        :param br: maximum number of bromine
        :param restrict:  boolean which if true indicates to use only CHNOPS
        :return: True if the formula is outside restrictions
        """
        if formula['C'] > c or formula['H'] > h or formula['N'] > n or formula['O'] > o or formula[
            'P'] > p or formula['S'] > s:
            return True
        else:
            return False


    def rule1(self, formula):
        """
        restrictions for the number of elements, from table 1 in 7 golden rules paper
        using the largest from the two sets, rather than a consecvent set
        :param formula: a formula
        :return: true if it passes the test
        """
        mass = self.get_formula_mass(formula)
        if mass < 500:
            if self.element_numbers_restriction(formula, 39, 72, 20, 20, 9, 10, 16, 10, 5): # c, h, n, o, p, s, f, cl, br
                return False
        elif mass < 1000:
            if self.element_numbers_restriction(formula, 78, 126, 25, 27, 9, 14, 34, 12, 8):
                return False
        elif mass < 2000:
            if self.element_numbers_restriction(formula, 156, 236, 32, 63, 9, 12, 48, 12, 10):
                return False
        elif mass < 3000:
            if self.element_numbers_restriction(formula, 162, 208, 48, 78, 6, 9, 16, 11, 8):
                return False
        return True


    def rule2(self, formula):
        """
        second rule regards valences and has 3 conditions:
        i. sum of all valences must be even
        ii. sum of all valences >= 2* max valence
        iii. sum of all valences >= 2 * #atoms-1
        second condition was not implemented
        :param formula: a formula
        :return: True if the formula passed this test
        """
        valence_sum = 0
        atoms = 0
        for element, number in formula.iteritems():
            valence_sum += ATOM_VALENCES[element] * number
            atoms += number
        if not (valence_sum % 2 == 0 and valence_sum >= 2 * (atoms - 1)):
            return False
        return True

    def rule3(self, formula):
        """
        not implemented, due to limited usefulness
        :return:
        """
        return True

    def rule4(self, formula):
        """
        rule 4 concerns the hydrogen to carbon ratios
        :param formula: a formula
        :return: True if the formula passed this test
        """
        if formula['C'] > 0 and formula['H'] > 0:
            h_c_ratio = (1.0*formula['H']) / (1.0*formula['C'])
            if not (6 > h_c_ratio > 0.1):
                return False
        return True

    def rule5(self, formula):
        """
        rule 5 concerns NOPS to carbon ratios
        :param formula: a formula
        :return: True if the formula passed this test
        """
        if formula['C'] > 0:
            n_c_ratio = formula['N'] / formula['C'] * 1.0
            o_c_ratio = formula['O'] / formula['C'] * 1.0
            p_c_ratio = formula['P'] / formula['C'] * 1.0
            s_c_ratio = formula['S'] / formula['C'] * 1.0
            if not (4 > n_c_ratio and 3 > o_c_ratio and 2 > p_c_ratio and 3 > s_c_ratio):
                return False
        return True

    def rule6(self, formula):
        """
        rule 6 concerns NOPS ratio among each other
        :param formula: a formula
        :return: True if the formula passed this test
        """
        n = formula['N']
        o = formula['O']
        p = formula['P']
        s = formula['S']
        if (n > 1 and o > 1 and p > 1 and s > 1 and not (n < 10 and o < 20 and p < 4 and s < 3)) or \
                (n > 3 and p > 3 and p > 3 and not (n < 11 and o < 22 and p < 6)) or \
                (o > 1 and p > 1 and s > 1 and not (o < 14 and p < 3 and s < 3)) or \
                (n > 1 and p > 1 and s > 1 and not (p < 3 and s < 3 and n < 4)) or \
                (n > 6 and o > 6 and s > 6 and not (n < 19 and o < 14 and s < 8)):
            return False
        return True


    def rule7(self, formula):
        """
        not implemented, due to limited usefulness
        :return:
        """
        return True
    
    def rule8(self, formula):
        """
        restrictions for the number of elements for C13, F and Cl
        :param formula: a formula
        :return: True if the formula passed this test
        """
        if self.rule_8_max_occurrences is None:
            max_occurrences = (1, 2, 2)
        else:
            max_occurrences = self.rule_8_max_occurrences
        if formula['C13'] > max_occurrences[0] or formula['F'] > max_occurrences[1] or formula['Cl'] > max_occurrences[2]:
            return False
        else:
            return True
    
    def make_formula_string(self, formula):
        f_string = ""
        for a in formula:
            if formula[a] == 1:
                f_string += a
            elif formula[a] > 1:
                f_string += "{}{}".format(a, formula[a])
        return f_string

    def filter_formula(self, formula):
        result = True
        breakdown = []
        if self.rule_switch[0]:
            out = self.rule1(formula)
            breakdown.append(out)
            result *= out
        if self.rule_switch[1]:
            out = self.rule2(formula)
            breakdown.append(out)
            result *= out
        if self.rule_switch[2]:
            out = self.rule3(formula)
            breakdown.append(out)
            result *= out
        if self.rule_switch[3]:
            out = self.rule4(formula)
            breakdown.append(out)
            result *= out
        if self.rule_switch[4]:
            out = self.rule5(formula)
            breakdown.append(out)
            result *= out
        if self.rule_switch[5]:
            out = self.rule6(formula)
            breakdown.append(out)
            result *= out
        if self.rule_switch[6]:
            out = self.rule7(formula)
            breakdown.append(out)
            result *= out
        if self.rule_switch[7]:
            out = self.rule8(formula)
            breakdown.append(out)
            result *= out

        return result, breakdown

    def filter_list(self, formula_list):
        filtered_formulas = []
        passed = []
        failed = {}
        for formula in formula_list:
            result,breakdown = self.filter_formula(formula)
            if result:
                filtered_formulas.append(formula)
                passed.append(self.make_formula_string(formula))
            else:
                failed[self.make_formula_string(formula)] = breakdown
        return filtered_formulas,passed,failed