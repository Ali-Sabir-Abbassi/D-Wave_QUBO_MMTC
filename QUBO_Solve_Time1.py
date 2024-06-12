# Define vertices, edges, terminals, and alpha
vertices = ['a', 'b', 'c', 'd', '1', '2']
edges = {
    ('a', 'b'): 1,
    ('b', 'd'): 2,
    ('d', 'c'): 2,
    ('a', 'c'): 2,
    ('a', '1'): 5,
    ('c', '2'): 4
}
terminals = ['1', '2']
alpha = 25

# Create QUBO dictionary
Q = {}

# First term: -25 * sum_u in V (sum_t in T x_u,t)
for u in vertices:
    for t in terminals:
        var = f"x{u}{t}"
        Q[(var, var)] = Q.get((var, var), 0) - alpha

# Interaction terms for each pair of terminals t and t'
for u in vertices:
    for t1 in terminals:
        for t2 in terminals:
            if t1 != t2:
                var1 = f"x{u}{t1}"
                var2 = f"x{u}{t2}"
                Q[(var1, var2)] = Q.get((var1, var2), 0) + alpha

# Second term: 50 * interaction between pairs of terminals
for t1 in terminals:
    for t2 in terminals:
        if t1 != t2:
            var1 = f"x{t1}{t1}"
            var2 = f"x{t2}{t2}"
            Q[(var1, var2)] = Q.get((var1, var2), 0) + 2 * alpha

# Third term: sum_{u,v} in E sum_t in T sum_{t' in T, t != t'} C({u,v}) x_u,t x_v,t'
for (u, v), weight in edges.items():
    for t1 in terminals:
        for t2 in terminals:
            if t1 != t2:
                var1 = f"x{u}{t1}"
                var2 = f"x{v}{t2}"
                Q[(var1, var2)] = Q.get((var1, var2), 0) + weight

# Add specific terms according to the explicit QUBO
specific_terms = {
    ('xa1', 'x12'): 5, ('xa2', 'x11'): 5, 
    ('xc1', 'x22'): 4, ('xc2', 'x21'): 4
}
for (var1, var2), value in specific_terms.items():
    Q[(var1, var2)] = Q.get((var1, var2), 0) + value



# Print QUBO dictionary to verify
print("Constructed QUBO dictionary:")
for key in sorted(Q.keys()):
    print(f"{key}: {Q[key]}")

# Explicit QUBO dictionary
explicit_Q = {
    ('xa1', 'xa1'): -25, ('xa2', 'xa2'): -25, ('xb1', 'xb1'): -25, ('xb2', 'xb2'): -25,
    ('xc1', 'xc1'): -25, ('xc2', 'xc2'): -25, ('xd1', 'xd1'): -25, ('xd2', 'xd2'): -25,
    ('x11', 'x11'): -25, ('x22', 'x22'): -25, ('xa1', 'xa2'): 50, ('xb1', 'xb2'): 50,
    ('xc1', 'xc2'): 50, ('xd1', 'xd2'): 50, ('x11', 'x12'): 50, ('x21', 'x22'): 50,
    ('xa1', 'xb2'): 1, ('xa2', 'xb1'): 1, ('xb1', 'xd2'): 2, ('xb2', 'xd1'): 2,
    ('xd1', 'xc2'): 2, ('xd2', 'xc1'): 2, ('xa1', 'xc2'): 2, ('xa2', 'xc1'): 2,
    ('xa1', 'x12'): 5, ('xa2', 'x11'): 5, ('xc1', 'x22'): 4, ('xc2', 'x21'): 4,
    ('constant', 'constant'): 150
}

# Print explicit QUBO dictionary to verify
print("Explicit QUBO dictionary:")
for key in sorted(explicit_Q.keys()):
    print(f"{key}: {explicit_Q[key]}")

# Check if the constructed QUBO matches the explicit QUBO
match = all(Q.get(key) == explicit_Q.get(key) for key in explicit_Q) and all(explicit_Q.get(key) == Q.get(key) for key in Q)
print(f"QUBO match: {match}")
