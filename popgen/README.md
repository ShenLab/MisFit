# Simulation of population genetics data
[simulation.py](simulation.py) simulates allele counts due to the effective population size in European history. 

[fiteur2.R](fiteur2.R) fits allele frequency distribution as a Gamma distribution with parameters as a function of selection coefficient and mutation rate. 

[est_sample_size.R](est_sample_size.R) MLE estimation of selection coefficient.

[sim_q/simulation_*.csv](sim_q/) The simulated allele counts distribution with given mutation rate. File names indicate the mutation rate. For each selection coefficient, there are 10,000 sites. 
Columns are: 1. mutation rate; 2. selection coefficient; 3. allele counts in the last generation; 4. number of occurrences in 10,000 sites. 
Column 3 divided by `2Ne` gives out allele frequency, where `Ne=3,386,437` is the population size of the last generation. 
