# Using these data splits

For the Banking-77, Clinic-150, and TREC/TREC-fine splits, this is fairly self-explanatory: load the dataset from Hugging Face and select these indices from the test set.

For NLU: There is no test split defined, so we first split the data into test and train using the `map-nlu-test` indices as the test set. Then, *from this test set*, we select a set of indices, described in `nlu-test-indices`.