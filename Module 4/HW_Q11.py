def q11(n):
    if n == 0:
        return 1
    else:
        return n + q11(n - 1)
        
def h1(n):
    return n + 1 + n * (n - 1) / 2.0
    
def h2(n):
    return 1.5 * n + .5
    
for i in range(8):
    print "i", i, q11(i), h1(i), h2(i)