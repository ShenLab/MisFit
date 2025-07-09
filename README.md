# MisFit

## A probabilistic graphical model for estimating selection coefficient of nonsynonymous variants from human population sequence data

Zhao et al 2025, https://www.nature.com/articles/s41467-025-59937-2

## MisFit version 1.5 data:
 * estimated S_gene for all human protein coding genes
 * estimated selection coefficient (MisFit S) for all possible missense variants in human genome caused by SNVs
Download at:  https://doi.org/10.5281/zenodo.15230898 
 * Note: In the original Misfit v1.5 implementation, 273 transcripts failed to map to canonical Ensembl IDs in the training data. We have now included these transcripts by directly using their non-canonical IDs. The additional scores are available at [here](https://zenodo.org/records/15851098?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImIzNzg4NzdkLTVjMjctNDAxMy1hNDVjLWM5ZjI2NDlkZmFhYiIsImRhdGEiOnt9LCJyYW5kb20iOiIzYjNhZDcxOTFlYmJiODIyOWE3NWI4YjJiYzEyN2UzNCJ9.krDcXXVI-8xW6k_LekKcDlgJ7BZxTihVAz_H4QqK84i1cT7CrLxxAJR7tZEo65i-GrnVIK5FNLQd2OU7UBLr_A)
 




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





