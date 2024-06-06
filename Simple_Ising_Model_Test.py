import dimod
import dwave.system

# Define a simple Ising model
bqm = dimod.BQM.from_ising({'a': -1, 'b': 1}, {('a', 'b'): 0.5})

# Set up the sampler that embeds our problem onto the D-Wave quantum computer
sampler = dwave.system.EmbeddingComposite(dwave.system.DWaveSampler())

# Sample 100 solutions
sampleset = sampler.sample(bqm, num_reads=100)

# Print the results
print(sampleset)