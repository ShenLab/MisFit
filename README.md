# MisFit
## predicting nonsynonymous variant selection coefficient 

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





