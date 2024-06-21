# In-context learning with long-context models 

This repository contains the code for reproducing the experiments from the preprint "In-Context Learning with Long-Context Models: An In-Depth Exploration." It is built on the skeleton of the code from the paper [Parallel Context Windows for Large Language Models
](https://arxiv.org/abs/2212.10947).  


## Use 

To run the experiments from the paper, use the `run_evaluation` script with appropriate arguments. For instance, to run ICL with several different amounts of data on Llama-80k:

```bash
python run_evaluation.py \
--dataset banking77 \
--model yaofu/llama-2-7b-80k \
--subsample-test-set 250 \
--n-runs 10 \
--seed 43 \
--nspw 1,10,25,100,1000,2000
--output-dir $OUTPUT_DIR
```

## Synchronizing data
We ran all final experiments on a 250-example test set sampled with seed=43. To ensure the exact same results, you may wish to use our precomputed test set (rather than re-sampling). The IDs for each test set are provided in the `dataset-splits` folder and are also visible by looking at the first column of any results file.

## Pre-computed outputs
If you do not wish to run inference, we also provide the outputs for all in-context learning experiments in the paper. The folder `final-results` contains experimental results, divided first by model and then by dataset. We provide the predicted label, true label, IDs of the examples used in the prompt, and label distribution of the prompt examples for reference. 

## Citation

If you use this code, please cite both In-Context Learning with Long-Context Models and the PCW paper:
```
@misc{bertsch2024incontext,
      title={In-Context Learning with Long-Context Models: An In-Depth Exploration}, 
      author={Amanda Bertsch and Maor Ivgi and Uri Alon and Jonathan Berant and Matthew R. Gormley and Graham Neubig},
      year={2024},
      eprint={2405.00200},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@misc{ratner2023parallel,
      title={Parallel Context Windows for Large Language Models}, 
      author={Nir Ratner and Yoav Levine and Yonatan Belinkov and Ori Ram and Inbal Magar and Omri Abend and Ehud Karpas and Amnon Shashua and Kevin Leyton-Brown and Yoav Shoham},
      year={2023},
      eprint={2212.10947},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
