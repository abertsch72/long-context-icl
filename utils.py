import logging
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
#from matplotlib import pyplot as plt
from numpy import typing as npt
from torch import distributed as dist
from transformers import PreTrainedTokenizerBase, LlamaTokenizer, LlamaTokenizerFast
from retriv import SparseRetriever

from constants import TEXT_BETWEEN_SHOTS, N_TOKENS, PROMPTS

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')


def get_max_n_shots(train_df: pd.DataFrame, test_df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase,
                    prompt_size: int) -> int:
    # this is nice info-- let's log this even if we don't need to use it 
    longest_test_prompt = test_df[N_TOKENS].max()
    _logger.info(f"longest_test_prompt = {longest_test_prompt}")

    n_tokens_between_shots = n_tokens_in_prompt(tokenizer, TEXT_BETWEEN_SHOTS)
    shot_lengths = train_df[N_TOKENS] + n_tokens_between_shots
    prompt_length_percentile = shot_lengths.quantile(0.9)
    print(f"Median length of demonstration: {shot_lengths.quantile(0.5)}")
    print(f"Mean length of demonstration: {sum(shot_lengths)/len(shot_lengths)}")

    max_possible_shots_length = prompt_size - longest_test_prompt
    return int(np.floor(max_possible_shots_length / prompt_length_percentile))

def retrieve_context(train_df: pd.DatetimeIndex, index: SparseRetriever, curr_example: str, n_examples: int, split_text, shuffle_seed=None):
    retrieved = index.search(
        query=curr_example,    # What to search for        
        return_docs=False,          # Default value, return the text of the documents
        cutoff=n_examples,                # Default value, number of results to return
    )
    inds = [int(d) for d in retrieved]
    
    if len(inds) < n_examples:
        print(f"WARNING: sampling {n_examples - len(inds)} examples randomly to fill window")
        inds.extend(train_df['id'].sample(n_examples - len(inds)))
    
    dps = list(train_df.loc[train_df['id'].isin(inds)]['prompts'])
    if shuffle_seed:
        import random
        prev_state = random.getstate()
        random.seed(shuffle_seed)
        random.shuffle(dps)
        random.setstate(prev_state)
        
    text = split_text.join(dps)
    return text

def create_retriever(train_df):
    sr = SparseRetriever(
        index_name="training-examples",
        model="bm25",
        min_df=1,
        tokenizer="whitespace",
        stemmer="english",
        stopwords="english",
        do_lowercasing=True,
        do_ampersand_normalization=True,
        do_special_chars_normalization=True,
        do_acronyms_normalization=True,
        do_punctuation_removal=True,
    )
    import random
    filename = f"__temp_index_file_{random.randint(1,5888)}_{random.randint(1,5999)}.csv"
    train_df['id'] = train_df.index
    from pathlib import Path
    import os
    if os.path.exists(filename):
        Path.unlink(Path(filename))
    
    train_df.to_csv(filename)
    sr.index_file(path=filename, 
        show_progress=True,  
        callback=lambda doc: {      # Callback defaults to None.
            "id": doc["id"],
            "text": doc["text"]},          
    )
    Path.unlink(Path(filename))

    return sr

def synchronize_examples_across_dfs(df1: pd.DataFrame, df2: pd.DataFrame, comp_column: str = "text"):
    df1 = df1.loc[df1[comp_column].isin(df2[comp_column])]
    df2 = df2.loc[df2[comp_column].isin(df1[comp_column])]
    return df1, df2

def filter_extremely_long_samples(df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase) -> pd.DataFrame:
    df[N_TOKENS] = df[PROMPTS].map(lambda x: n_tokens_in_prompt(tokenizer, x))
    mask = df[N_TOKENS] <= df[N_TOKENS].quantile(0.99)
    _logger.info(f"filtered {sum(~mask)} from  dataset due to extreme length")
    df = df.loc[mask].copy()
    _logger.info(f"longest remaining prompt according to tokenizer: {df[N_TOKENS].max()}")
    return df


def n_tokens_in_prompt(tokenizer: PreTrainedTokenizerBase, prompt: str, add_special_tokens=False) -> int:
    return len(tokenizer.encode(prompt, add_special_tokens=add_special_tokens))


def plot_results_graph(results, dataset_name, n_shots, model='') -> None:
    plt.figure()
    plt.errorbar(n_shots, np.mean(results, axis=1), np.std(results, axis=1), fmt='*')
    plt.xlabel("# shots")
    plt.xticks(n_shots)
    metric = 'Accuracy'
    plt.ylabel(f"{dataset_name} {metric}")
    plt.title(f"{metric} {dataset_name} {model}")


def load_results(dataset_name: str, output_dir: str, plot=False) -> Tuple[npt.NDArray[float], List[int]]:
    all_results = os.listdir(output_dir)
    results_path = [r for r in all_results if r.startswith(f'{dataset_name}_')]
    if len(results_path) != 1:
        raise ValueError(f"Found {len(results_path)} results!")
    results_path = results_path[0]
    results = np.load(os.path.join(output_dir, results_path))
    n_shots = [int(d) for d in results_path.split('.')[-2].split('_') if d.isdigit()]
    if plot:
        plot_results_graph(results, dataset_name, n_shots)
    return results, n_shots

def save_results(dataset: str, n_shots: List[int], results: np.ndarray[int], predictions: List[str], outpath: str,
                 model: str = '', plot_results: bool = True) -> None:
    if plot_results:
        plot_results_graph(results, dataset, n_shots, model)
        plt.show()
    if not dist.is_initialized() or dist.get_rank() == 0:
        # in case we use multiple GPUs - we only save one file
        np.save(outpath, results)
        with open(outpath.split(".")[0] + "-outputs.pkl", 'wb') as f:
            import pickle
            pickle.dump(predictions, f)
        clean_name = outpath.split(".")[0].split('/')[-1]
        for num, nshots in enumerate(n_shots):
            for i, rep in enumerate(predictions[num]):
                # need to add id and output columns 
                rep['id'] = rep.index
                rep['n_shots'] = nshots
                rep['run_number'] = i
                if "nspw" in clean_name: # nspw={nspw}_
                    rep["nspw"] = clean_name.split("nspw=")[-1].split("_")[0]
                with open(os.path.dirname(outpath) + "/" + clean_name.split("n_shots_")[0]+"+n_shots="+str(nshots)+"+run="+str(i)+".csv", 'w') as f:
                    rep.to_csv(f)

def encode_labels(tokenizer: PreTrainedTokenizerBase, labels: List[str]) -> List[List[int]]:
    if isinstance(tokenizer, LlamaTokenizer):
        # sentence piece - adds a space at the beginning of the sentence
        return [tokenizer.encode(f'{label.lstrip()}', add_special_tokens=False) for label in labels]

    return [tokenizer.encode(f' {label.lstrip()}', add_special_tokens=False) for label in labels]


def encode_stop_seq(tokenizer: PreTrainedTokenizerBase, stop_seq: str) -> int:
    stop_seq_token_id = tokenizer.encode(stop_seq, add_special_tokens=False)
    if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
        assert len(stop_seq_token_id) == 2
    else:
        assert len(stop_seq_token_id) == 1
    return stop_seq_token_id[-1]
