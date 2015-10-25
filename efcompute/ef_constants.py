import sys

INFINITE = sys.maxint

PROTON_MASS = 1.00727645199076
ATOM_NAME_LIST = ['C', 'H', 'N', 'O', 
                  'P', 'S', 'C13', 'F', 
                  'Cl']

ATOM_MASS_LIST = [12.00000000000,   1.00782503214,    14.00307400524,   15.99491462210, 
                  30.97376151200,   31.97207069000,   13.00335483780,   18.99840325000, 
                  34.96885271000]
ATOM_VALENCE_LIST = [4, 1, 3, 2, 3, 2, 4, 1, 1]

ATOM_MASSES = dict(zip(ATOM_NAME_LIST, ATOM_MASS_LIST))
ATOM_VALENCES = dict(zip(ATOM_NAME_LIST, ATOM_VALENCE_LIST))

# all the rules are turned on by default
DEFAULT_RULES_SWITCH = [True, True, True, True, True, True, True, True]

RULE_8_MAX_OCCURRENCES = {}
for name in ATOM_NAME_LIST:
    RULE_8_MAX_OCCURRENCES[name] = INFINITE