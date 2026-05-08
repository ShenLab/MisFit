# MisFit

## A probabilistic graphical model for estimating selection coefficient of nonsynonymous variants from human population sequence data

Zhao et al 2025, https://www.nature.com/articles/s41467-025-59937-2

## MisFit version 1.5.1 data:
 * estimated S_gene for all human protein coding genes
 * estimated selection coefficient (MisFit S) for all possible missense variants in human genome caused by SNVs: [MisFit v1.5.1](https://github.com/ShenLab/MisFit/releases/download/v1.5.1/MisFit_S_SNV_v1.5.1.txt.gz)
 * Note: In Misfit v1.5 released with Zhao et al 2025, 273 transcripts failed to map to canonical Ensembl IDs in the training data. We have patched these transcripts by directly using their non-canonical IDs in v1.5.1

### population genetics model
[pop_model](pop_model_2)
simulate variants and construct PIG model

### protein-truncating variants
[model_PTV](model/model_PTV) only use PTVs, independent of other models

### prior of missense variants
[model_mis](model/model_mis)
used to find priors of `d` and `s_gene`, then initialize `s_gene` before MisFit training. 

### Baseline models
[model_basic](model/model_basic) population data w./w.o. genes

[model_logit](model/model_logit) population data + gene + ESM zero-shot as `d`

### MisFit model
[model_TF](model/model_TF) full MisFit model

`*_analysis` are used to combine data for different analysis

Note: `model_selection` directly given by the model may need to be transformed by a sigmoid function to get MisFit_S in the original scale

### evaluation and figure-plotting
[model_evaluate](model_evaluate)

### data processing
to be updated
- deep mutational scan GMM
- variant annotations





