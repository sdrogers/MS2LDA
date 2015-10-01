import sys
import fractions
# Set infinity
infinite = sys.maxint


def find_n(n, p, d):
    minimum = infinite
    for q, elem in enumerate(n):
        if q % d == p:
            if elem < minimum:
                minimum = elem
    return minimum


def round_robin(a):
    # file_handler = open(filename, 'w')
    # file_handler.write("Precision: " + str(b) + '\n')

    # initialize the variables
    k = len(a)
    a1 = a[0]
    n = [infinite for r in range(0, a1)]
    n[0] = 0
    

    rr = []
    rr.append(list(n))
    # file_handler.write(str(n) + '\n')

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
        # print n
        rr.append(list(n))
        # file_handler.write(str(n) + '\n')

    return rr
    
    # file_handler.close()


# def parse_rt(filein):
#     with open(filein, 'r') as f:
#         data_in = f.read()
#     f.close()
#     rows = data_in.splitlines()
#     # set the same blowup factor
#     b = int(rows[0].split()[1])

#     parsed_rows = []
#     for row in rows[1:]:
#         row = row.split('[')[1].split(']')[0].split(',')
#         parsed_rows.append([int(item) for item in row])
#     return b, parsed_rows

def find_all_simon(mass,i,c,a,rt):
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
                find_all_simon(m,i-1,c,a,rt)
                m = m - lcm
                c[i] = c[i] + l

def find_all(mass_in, mass, i, c, a, rt):
    if i == -1:
        # c[0] = mass / a[0]
        # formula = {elem: c[i] for i, elem, in enumerate(sorted_CHNOPS())}
        # formula_mass = get_formula_mass(formula)
        
        su = sum([e*c[i] for i,e in enumerate(a)])
        if su == mass_in:
            formula = {e : c[i] for i,e in enumerate(a)}
            formulas.append(formula)

        # if abs(formula_mass - mass_in) < 1:
        #     formulas.append(formula)

    else:
        lcm = a[0] * a[i] / fractions.gcd(a[0],a[i])
        l = lcm / a[i]
        for j in range(0, l):
            c[i] = j
            m = mass - j * a[i]
            r = m % a[0]
            lbound = rt[i-1][r]
            while m >= lbound:
                find_all(mass_in, m, i-1, c, a, rt)
                m = m - lcm
                c[i] = c[i] + l



if __name__=='__main__':
    global formulas
    formulas = []
    a = [5,8,9,12]
    rr = round_robin(a)
    # b,rr2 = parse_rt('test.txt')
    print rr
    k = len(rr)
    c = [0 for i in range(0, k)]
    mass_int = int(sys.argv[1])
    # find_all(mass_int,mass_int,k-1,c,a,rr)
    find_all_simon(mass_int,k-1,c,a,rr)
    print 'found ' + str(len(formulas))
    print formulas
