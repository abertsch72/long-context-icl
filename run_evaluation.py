import argparse
import logging
from typing import List, Optional

import pandas as pd
from transformers import PreTrainedTokenizerBase
import numpy as np

from transformers import LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM

from datasets_loader import DATASET_NAMES2LOADERS, TEST_ON_TRAIN_DATASETS, get_loader
from experiment_manager import ExperimentManager
from model_loaders import load_pcw_wrapper
from utils import get_max_n_shots, filter_extremely_long_samples, save_results
import os

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')


def get_dataset(dataset: str, tokenizer: PreTrainedTokenizerBase, token=None, half_seed=None) -> (pd.DataFrame, pd.DataFrame, List):
    da = get_loader(dataset, token, half_seed)
    # Filter extremely long samples from both train and test samples:
    _logger.info("filtering test set:")
    test_df = filter_extremely_long_samples(da.test_df, tokenizer)
    _logger.info("filtering train set:")
    train_df = filter_extremely_long_samples(da.train_df, tokenizer)
    
    is_classification = set(da.labels).issubset(da.train_labels)
    return test_df, train_df, da.labels, is_classification

def run_pcw_experiment(datasets: List[str], models: List[str], cache_dir: str, subsample_test_set: int, output_dir: str,
                       n_windows: List[int], n_shots_per_window: List[int], n_runs: int,
                       random_seed: int, right_indentation: bool, token=None, overwrite=False,
                       pcw_base_model=False, fp16=False, stride_size=-1, examples_stride=-1, 
                       half_seed=None, window_shuffle_seed=None, use_retrieval=False, sort_by_label = False) -> None:
    assert not (window_shuffle_seed and sort_by_label), "can't sort and shuffle!"
    
    base_output_dir = output_dir
    all_records = []
    for model in models:
        clean_model_name = model.replace('/', '+').replace(' ', '_')
        if stride_size != -1:
            clean_model_name += f"_stride_size={stride_size}"
        if examples_stride != -1:
            clean_model_name += f"_examples_stride={examples_stride}"
        
        print(f'* Starting with model: {model} ({clean_model_name})')
        pcw_model = None

        for dataset in datasets: 

            clean_dataset_name = dataset.replace('/', '+').replace(' ', '_')
            if half_seed:
                print('Using half label space; renamed dataset to avoid confusion')
                clean_dataset_name = f"{clean_dataset_name}-half-seed={half_seed}"
                print(f"New dataset name: {clean_dataset_name}")
            if window_shuffle_seed:
                print('Reshuffling examples in-window; renamed dataset to avoid confusion')
                clean_dataset_name = f"{clean_dataset_name}-window-shuffle-seed={window_shuffle_seed}"
                print(f"New dataset name: {clean_dataset_name}")
            if use_retrieval:
                print('Retrieving examples in-window; renamed dataset to avoid confusion')
                clean_dataset_name = f"{clean_dataset_name}-retrieval"
                print(f"New dataset name: {clean_dataset_name}")
            if sort_by_label:
                clean_dataset_name = f"{clean_dataset_name}-sorted"
                
            print(f'\t- Running with dataset: {dataset} ({clean_dataset_name})')
            test_on_train = (dataset in TEST_ON_TRAIN_DATASETS)
            output_dir = os.path.join(base_output_dir, clean_model_name, clean_dataset_name)

            test_df, train_df, labels = None, None, None

            records = []

            for nspw in n_shots_per_window:
                n_shots = [i * nspw for i in n_windows]  # here nspw may still be -1
                
                output_str = ""
                if nspw > 1:
                    output_str += f"nspw={nspw}_"
                output_path = os.path.join(output_dir, f"{output_str}n_shots_results_{'_'.join([str(i) for i in n_shots])}.npy")
                nshots_file_name = os.path.join(output_dir, f"nspw={nspw}-n_shots.txt")

                # TODO - incorporate n_runs in the caching system, so we can easily add additional runs, without running from scratch (or get different number of runs)
                # TODO - also, the name currently contains the number of windows to have, so it's impossible to add more windows and use cache, just more nspw
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                if os.path.exists(output_path) and not overwrite:
                    print(f'Found results in {output_path}. Skipping...')
                    
                    # just load the records before
                    # load the npy file
                    accuracies = np.load(output_path)
                    with open(nshots_file_name, 'r') as f:
                        n_shots = eval(f.read())
                    rows, cols = accuracies.shape
                    for i in range(rows):
                        for j in range(cols):
                            record = {
                                "n_shots": n_shots[i],
                                "accuracy": accuracies[i][j],
                                "run_num": j,
                                "nspw": nspw
                            }
                            records.append(record)

                    continue
                elif os.path.exists(output_path) and overwrite:
                    print(f'Found results in {output_path}. Overwriting...')
                else:
                    print(f'Running with {output_path}...')

                if pcw_model is None:
                    # lazy loading

                    if "claude" in model:
                        pcw_model = "claude"
                        tokenizer = None
                        context_window_size = 200_000
                    elif "80k" in model:
                        import torch

                        def reset_rope(model, model_max_train_len, scaling_factor):
                            for l in model.model.layers:
                                l.self_attn.rotary_emb.scaling_factor = scaling_factor
                                l.self_attn.rotary_emb._set_cos_sin_cache(seq_len=model_max_train_len, device="cpu", dtype=torch.float32)
                            return
                        pcw_model = AutoModelForCausalLM.from_pretrained(model,
                                                                    #use_flash_attention_2="flash_attention_2", 
                                                                    torch_dtype=torch.bfloat16
                                                                    ).to('cuda') # requires about 14G disk size in $HF_HOME
                        scaling_factor = 10 # hardcode here
                        context_window_size = 81920
                        reset_rope(pcw_model, model_max_train_len=context_window_size, scaling_factor=scaling_factor)
                        pcw_model.eval()
                        
                        tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=token)

                        
                    elif not pcw_base_model and len(n_windows) == 1 and n_windows[0] == 1:
                        if 'llama' in model.lower():
                            pcw_model = LlamaForCausalLM.from_pretrained(model, use_auth_token=token).to('cuda')
                        else:
                            pcw_model = AutoModelForCausalLM.from_pretrained(model, use_auth_token=token).to('cuda')
                            print('Warning - you are using a non-llama model which may make some of the features to not work correctly. beware')
                        tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=token)
                        if fp16:
                            pcw_model.half()
                        pcw_model.eval()
                        context_window_size = pcw_model.config.max_position_embeddings # for llama
                    else:
                        pcw_model = load_pcw_wrapper(model, cache_dir, right_indentation, max(n_windows), token=token)
                        tokenizer = pcw_model.tokenizer
                        context_window_size = pcw_model.context_window_size

                    print('Loaded model')
                
                if test_df is None:
                    # lazy loading
                    test_df, train_df, labels, is_classification = get_dataset(dataset, tokenizer, token=token, half_seed=half_seed)
                    print('Loaded dataset')
                    if half_seed:
                        print('Using half label space; renaming dataset (part 2!)')
                        dataset = f"{dataset}-half-seed={half_seed}"


                if nspw == -1:
                    # default behavior: we take the maximum number of samples per window
                    nspw = get_max_n_shots(train_df, test_df, tokenizer, context_window_size)
                    _logger.info(f"Found max n shot per window = {nspw}")
                print(f'Running with NSPW={nspw}, and n_windows={n_windows}')

                n_shots = [i * nspw for i in n_windows]
                with open(nshots_file_name, 'w') as f:
                    f.write(str(n_shots))

                em = ExperimentManager(test_df, train_df, pcw_model, tokenizer=tokenizer, labels=labels, random_seed=random_seed,
                                    n_shots_per_window=nspw, subsample_test_set=subsample_test_set, test_on_train = test_on_train,
                                    pcw_base_model=pcw_base_model, context_size=context_window_size, 
                                    is_classification=is_classification, stride_size=stride_size, examples_stride=examples_stride,
                                    window_shuffle_seed = window_shuffle_seed, use_retrieval=use_retrieval, sort_by_label=sort_by_label)

                accuracies, predictions = em.run_experiment_across_shots(n_shots, n_runs,context_window_size=context_window_size)  # an ndarry of shape (n_runs, len(n_shots))
                
                save_results(dataset, n_shots, accuracies, predictions, output_path, model, plot_results=False)

                rows, cols = accuracies.shape

                for i in range(rows):
                    for j in range(cols):
                        record = {
                            "n_shots": n_shots[i],
                            "accuracy": accuracies[i][j],
                            "run_num": j,
                            "nspw": nspw
                        }
                        records.append(record)

            # assume output dir already contains the model name
            fname = f"{output_dir}/n_shots_results_over_{subsample_test_set}_samples_seed_{random_seed}_ri={right_indentation}.csv"
            pd.DataFrame(records).to_csv(fname, index=False)
            print('---------------------------------------------------')
            print(f'Done running model {model} on dataset {dataset}. You can find the results in {fname}')
            
            all_records.extend([r | {'model': model, 'dataset': dataset} for r in records])  # require python 3.9+
    fname = f"{base_output_dir}/all_results_over_{subsample_test_set}_samples_seed_{random_seed}_ri={right_indentation}.csv"
    pd.DataFrame(all_records).to_csv(fname, index=False)
    print('---------------------------------------------------')
    print(f'Done running all models on all datasets. You can find the results in {fname}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Datasets and model related arguments
    parser.add_argument('--datasets', nargs='+', 
                        help=f'Name of datasets. Supported datasets: {DATASET_NAMES2LOADERS.keys()}')
    parser.add_argument('--models', nargs='+',
                        help='HF model names to use, either gpt2 or LLaMa family models')
    parser.add_argument('--fp16', help="use half precision",
                        action='store_true', default=False)
    
    # Directories, caching, and I/O arguments
    parser.add_argument('--output-dir', help="Directory for saving the results", default='./temp', type=str)
    parser.add_argument('--cache-dir', help="Hugging face cache dir", type=str, default=None)
    parser.add_argument('--token', default=None, type=str, help='HF token if needed')
    parser.add_argument('--overwrite', help="If true, overwrite existing results", action='store_true', default=False)
    
    # Evaluation and sampling related arguments
    parser.add_argument('--subsample-test-set', type=int,
                        help='Size of test set to use to speed up eval. None means using all test set.')
    parser.add_argument('--random-seed', default=42, type=int)
    parser.add_argument('--n-runs', help="Number of times experiments are repeated for every number of windows",
                        type=int, default=1)

    # Windowing related arguments
    parser.add_argument('-n', '--n-windows', nargs='+', help="Number of parallel context windows", type=int)
    parser.add_argument('--n-shots-per-window', nargs='+',
                        help="number of examples to fit in each window (can be multiple items). Use -1 for maximum possible",
                        type=int, required=True)
    parser.add_argument('--right-indentation', help="indent all windows to the right",
                        action='store_true', default=False)
        
    parser.add_argument('--stride-size', help="how much of the input to allow to attend to each other",
                        type=int, required=False, default=-1)
        
    parser.add_argument('--examples-stride', help="how much of the input to allow to attend to each other (in # of examples)",
                        type=int, required=False, default=-1)
        
    parser.add_argument('--half-seed', help="if subsampling label space, a seed to use",
                        type=int, required=False, default=None)
    parser.add_argument('--window-shuffle-seed', help="if re-shuffling examples in window, a seed to use",
                        type=int, required=False, default=None)
    parser.add_argument('--sort-by-label', help="sort examples by label",
                        action='store_true', required=False, default=False)

    parser.add_argument('--use-retrieval', help="apply retrieval method",
                        action='store_true', default=False)
    
    parser.add_argument('--pcw-base-model', help="when we have only 1 window, use PCW code (if not set: will use basic inference path instead)",
                        action='store_true', default=False)

    
    args = parser.parse_args()
    
    print('running with token:', args.token)
    run_pcw_experiment(**vars(args))
    # Windowing related arguments
