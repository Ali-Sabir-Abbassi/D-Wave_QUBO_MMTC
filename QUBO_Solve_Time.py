import time
from minorminer import find_embedding
from dwave.system import DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite
from dimod import BinaryQuadraticModel, Vartype

# Define vertices and edges
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

# First term: 25 * sum_u in V ((1 - sum_t in T x_u,t)^2)
for u in vertices:
    for t in terminals:
        var = f"x{u}{t}"
        Q[(var, var)] = Q.get((var, var), 0) + alpha
        for t2 in terminals:
            if t != t2:
                var2 = f"x{u}{t2}"
                Q[(var, var2)] = Q.get((var, var2), 0) - alpha

# Second term: 25 * sum_t in T sum_t' in T, t != t' x_t,t'
for t1 in terminals:
    for t2 in terminals:
        if t1 != t2:
            var1 = f"x{t1}{t1}"
            var2 = f"x{t2}{t2}"
            Q[(var1, var2)] = Q.get((var1, var2), 0) + alpha

# Third term: sum_{u,v} in E sum_t in T sum_{t' in T, t != t'} C({u,v}) x_u,t x_v,t'
for (u, v), weight in edges.items():
    for t1 in terminals:
        for t2 in terminals:
            if t1 != t2:
                var1 = f"x{u}{t1}"
                var2 = f"x{v}{t2}"
                Q[(var1, var2)] = Q.get((var1, var2), 0) + weight

# Convert QUBO dictionary to Binary Quadratic Model (BQM)
bqm = BinaryQuadraticModel.from_qubo(Q)

# Setup D-Wave sampler
token = 'DEV-931589de6c7212e1a045256e9107208d3be142cd'
sampler = DWaveSampler(token=token)

# Extract target edgelist and adjacency from sampler
_, target_edgelist, target_adjacency = sampler.structure

# Function to calculate embedding
def calculate_embedding(Q_dict, target_edgelist):
    start = time.time()
    emb_precalc = find_embedding(Q_dict, target_edgelist, verbose=1)
    end = time.time()
    emb_precalc_dt = end - start
    return emb_precalc, emb_precalc_dt

# Calculate the embedding locally
Q_dict, Q_const = bqm.to_qubo()
emb_precalc, emb_precalc_dt = calculate_embedding(Q_dict, target_edgelist)

# Setup Fixed Embedding Composite with the precomputed embedding
emb = FixedEmbeddingComposite(sampler, emb_precalc)

# Run the QUBO on D-Wave
num_reads = 100
job_label = 'QUBO_Example'
sample_start = time.time()
response = emb.sample(bqm, num_reads=num_reads, label=job_label)
sample_end = time.time()
sample_dt = sample_end - sample_start

# Extract and print the solution
solution = response.first.sample
energy = response.first.energy
print(f"Solution: {solution}")
print(f"Energy: {energy}")
print(f"Embedding calculation time: {emb_precalc_dt}")
print(f"Sampling time: {sample_dt}")
