# PIG population genetics model
[Durbin_population_history.xlsx](Durbin_population_history.xlsx) sheet 1 is from the supplementary of original Schiffels and Durbin 2014, sheet 2 is the combined Ne picked from different submodel, which is used for generating european effective population size history. (30 years per generation)
[EUR_pop.csv](EUR_pop.csv) adjusted EUR Ne, used for simulation
[generate_history.py](generate_history.py) generate EUR Ne history, setting a final Ne
[simulation_syn.py](simulation_syn.py) simulate synonymous variants
[simulation_grid.py](simulation_grid.py) simulate variants with different mutation rate and selection coefficient
[syn_summary.py](syn_summary.py) summarize synonymous variants and calulate kl
[sim_q](sim_q) simulated allele frequency distribution by mutation rate
[fiteur_PIG.R](fiteur_PIG.R) fit PIG distribution from simulated results
[plot_IG_pars.R](plot_IG_pars.R) plot PIG parameters
[plot_PIG.R](plot_PIG.R) plot likelihoods
[est_agg.R](est_agg.R) estimate categorical s

