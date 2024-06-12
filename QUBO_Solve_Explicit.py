import time
from minorminer import find_embedding
from dwave.system import DWaveSampler, FixedEmbeddingComposite
from dimod import BinaryQuadraticModel

# Explicit QUBO definition
Q = {
    ('xa1', 'xa1'): -25, ('xa2', 'xa2'): -25, ('xb1', 'xb1'): -25, ('xb2', 'xb2'): -25,
    ('xc1', 'xc1'): -25, ('xc2', 'xc2'): -25, ('xd1', 'xd1'): -25, ('xd2', 'xd2'): -25,
    ('x11', 'x11'): -25, ('x22', 'x22'): -25, ('xa1', 'xa2'): 50, ('xb1', 'xb2'): 50,
    ('xc1', 'xc2'): 50, ('xd1', 'xd2'): 50, ('x11', 'x12'): 50, ('x21', 'x22'): 50,
    ('xa1', 'xb2'): 1, ('xa2', 'xb1'): 1, ('xb1', 'xd2'): 2, ('xb2', 'xd1'): 2,
    ('xd1', 'xc2'): 2, ('xd2', 'xc1'): 2, ('xa1', 'xc2'): 2, ('xa2', 'xc1'): 2,
    ('xa1', 'x12'): 5, ('xa2', 'x11'): 5, ('xc1', 'x22'): 4, ('xc2', 'x21'): 4
}

# Add the constant term to an arbitrary variable to ensure it's considered in the energy calculation
Q[('constant', 'constant')] = 150

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
emb_precalc, emb_precalc_dt = calculate_embedding(Q, target_edgelist)

# Setup Fixed Embedding Composite with the precomputed embedding
emb = FixedEmbeddingComposite(sampler, emb_precalc)

# Run the QUBO on D-Wave
num_reads = 1000
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

# Verify the result with expected solution
expected_solution = {
    'x11': 1, 'x12': 0, 'x21': 0, 'x22': 1,
    'xa1': 1, 'xa2': 0, 'xb1': 0, 'xb2': 1,
    'xc1': 0, 'xc2': 1, 'xd1': 0, 'xd2': 1
}

correct = all(solution.get(key, 0) == expected_solution[key] for key in expected_solution)
print(f"Solution matches expected: {correct}")
