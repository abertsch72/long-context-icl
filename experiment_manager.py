import logging
import random
from typing import List, Dict
from collections import Counter
from typing import Optional, Union

import numpy as np
import torch
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm

from constants import TEXT_BETWEEN_SHOTS, N_TOKENS, PROMPTS
from datasets_loader import LABEL_TOKENS
from pcw_wrapper import PCWModelWrapper

from logits_processor import RestrictiveTokensLogitsProcessor
from utils import n_tokens_in_prompt, encode_labels, encode_stop_seq, synchronize_examples_across_dfs, retrieve_context, create_retriever

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

STOP_SEQUENCE = '\n'


class ExperimentManager:
    def __init__(self, test_df: pd.DataFrame, train_df: pd.DataFrame, model: PCWModelWrapper, tokenizer,
                 labels: List[str], random_seed: int = 42, subsample_test_set: int = 250,
                 n_shots_per_window: int = None, test_on_train: bool=False,
                 pcw_base_model: bool = False, context_size: int = 4096, is_classification: bool = True,
                 stride_size: int = -1, examples_stride: int = -1, window_shuffle_seed: int = None,
                 use_retrieval: bool = False, sort_by_label=False):

        if test_on_train:                 # remove examples from train and test if they're only in one 
            train_df, test_df = synchronize_examples_across_dfs(train_df, test_df)

        if subsample_test_set < len(test_df):
            if test_on_train:
                _logger.warning("You're testing on train data; only do this as a sanity check!")
                self.full_test_df = test_df.copy()
            else:
                np.random.seed(random_seed)
                test_df = test_df.sample(subsample_test_set)
        
        self.subsample_test_set = subsample_test_set
        self.sync_shuffle = test_on_train
        self.test_df = test_df
        self.train_df = train_df
        self.model = model
        self.base_random_seed = random_seed
        self.n_shots_per_window = n_shots_per_window
        self.tokenizer = tokenizer
        self.label_distribution_prompt = dict()
        self.label_map = None
        self.reverse_label_map = None
        self.is_classification = is_classification
        self.pcw_base_model = pcw_base_model
        self.context_size = context_size
        self.stride_size = stride_size 
        self.examples_stride = examples_stride
        self.window_shuffle_seed = window_shuffle_seed
        self.use_retrieval = use_retrieval
        self.sort_by_label = sort_by_label
        
        np.random.seed(random_seed)
        self.random_orders = [np.random.permutation(list(self.train_df.index)) for i in range(20)]
        self.times_shuffled = 0
        if is_classification:
            _logger.info(f"Setting up labels and logt processor for {len(labels)} possible labels")
            self._initialize_labels_and_logit_processor(labels)
        else:
            self._fix_labels_wrt_tokneizer(labels)
            self.logit_processor = None

    def synchronize_examples():
        pass
    
    def _initialize_labels_and_logit_processor(self, labels: List[str]) -> None:
        map_labels, pad, shorten_label_tokens = self._fix_labels_wrt_tokneizer(labels)
        labels_tokens_array = np.array(
            [i + [self.tokenizer.eos_token_id] * (pad - len(i)) for i in shorten_label_tokens])
        labels_tokens_array = self.pad_contained_labels_with_stop_seq(shorten_label_tokens, labels_tokens_array)
        self.logit_processor = RestrictiveTokensLogitsProcessor(restrictive_token_ids=labels_tokens_array,
                                                                eos_token_id=self.tokenizer.eos_token_id)
        self.possible_labels = set(map_labels.values())

    def _fix_labels_wrt_tokneizer(self, labels):
        _logger.info(f"Provided labels: {labels}")
        labels_tokens = encode_labels(self.tokenizer, labels)
        labels_tokens_array = self.minimize_labels_tokens(labels_tokens)
        _logger.info(f"Provided labels average n_tokens: {np.round(np.mean([len(lt) for lt in labels_tokens]), 3)}")
        # we fix the labels accordingly in the test set:
        shorten_label_tokens = [t[t != self.tokenizer.eos_token_id].tolist() for t in labels_tokens_array]
        _logger.info(
            f"shortened labels average n_tokens: {np.round(np.mean([len(lt) for lt in shorten_label_tokens]), 3)}")
        # Moving the test set label tokens to their shorter version:
        map_labels = {old_label: self.tokenizer.decode(t).lstrip() for old_label, t in
                      zip(labels, shorten_label_tokens)}
        self.label_map = lambda a: map_labels[a]
        inv_map = {v: k for k, v in map_labels.items()}
        self.reverse_label_map = inv_map
        self.test_df[LABEL_TOKENS] = self.test_df[LABEL_TOKENS].map(map_labels)
        pad = len(max(shorten_label_tokens, key=len))
        self.max_n_tokens = pad

        return map_labels, pad, shorten_label_tokens

    def minimize_labels_tokens(self, labels_tokens: List[List[int]]) -> npt.NDArray[int]:
        """
         Minimize the number of tokens per label to be the shortest possible unique one.
        """
        pad = len(max(labels_tokens, key=len))
        labels_tokens_array = np.array([i + [self.tokenizer.eos_token_id] * (pad - len(i)) for i in labels_tokens])
        for i, tokens in enumerate(labels_tokens):
            for j in range(len(tokens)):
                labels_with_shared_beginnings = np.sum(
                    np.all(labels_tokens_array[:, :j] == np.array(tokens[:j]), axis=1))
                if labels_with_shared_beginnings == 1:
                    labels_tokens_array[i, j:] = self.tokenizer.eos_token_id
                    break
        return labels_tokens_array

    def pad_contained_labels_with_stop_seq(self, labels_tokens: List, labels_tokens_array: npt.NDArray[int]) -> npt.NDArray[int]:
        """
        In case we have two labels, where one label contains the other label (for example: "A" and "A B") we need
        to allow the restrictive decoding to produce the output "A". We support it by adding "\n" to the shorter label.
        """
        stop_seq_token_id = encode_stop_seq(self.tokenizer, STOP_SEQUENCE)
        for i, tokens in enumerate(labels_tokens):
            labels_with_shared_beginnings = np.sum(
                np.all(labels_tokens_array[:, :len(tokens)] == np.array(tokens), axis=1))
            if labels_with_shared_beginnings > 1:
                _logger.info(f"label{self.tokenizer.decode(tokens)} is the beginning of one of the other labels,"
                             f"adding stop sequence to its end")
                labels_tokens_array[i, len(tokens)] = stop_seq_token_id
        return labels_tokens_array

    def _set_random_seed(self, random_seed: int) -> None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    def get_few_shots_acc(self, windows_few_shot: List[str]) -> float:
        if self.use_retrieval:
            predicted_labels = self.get_predicted_labels_retrieval(restrictive_logit_preprocessor=self.logit_processor)
        elif len(windows_few_shot) == 1 and not self.pcw_base_model:
            predicted_labels = self.get_predicted_labels_nonpcw(context=windows_few_shot[0], restrictive_logit_preprocessor=self.logit_processor)
        else:
            predicted_labels = self.get_predicted_labels(windows_few_shot)
        return self.calc_acc(predicted_labels, windows_few_shot[0])

    def make_block_mask(self, fewshot_examples, context):
        fewshot_len = fewshot_examples['input_ids'].shape[-1]

        splitter = self.tokenizer(TEXT_BETWEEN_SHOTS, add_special_tokens=False, return_tensors='pt')['input_ids']
        splitter_list = self.tokenizer(TEXT_BETWEEN_SHOTS, add_special_tokens=False)['input_ids']
        
        context_examples = context.split(TEXT_BETWEEN_SHOTS)
        fewshot_examples_new = []
        example_boundaries = [0]
        for i, ex in enumerate(context_examples):
            if i == 0:
                fewshot_examples_new.extend(self.tokenizer(ex, add_special_tokens=False)['input_ids']) 

            else:
                fewshot_examples_new.extend(self.tokenizer(TEXT_BETWEEN_SHOTS + ex, add_special_tokens=False)['input_ids'][1:]) 
            example_boundaries.append(len(fewshot_examples_new))
            
            
        fewshot_examples_new = torch.tensor(fewshot_examples_new).unsqueeze(0)
        assert fewshot_examples['input_ids'].equal(fewshot_examples_new)
        fewshot_examples['input_ids'] = fewshot_examples_new


        # Get the indices where matches occur
        starting_positions = example_boundaries[:-1]
        
        
        if self.stride_size != -1:
            starting_positions = torch.tensor(starting_positions)
            ideal_block_indices = torch.arange(self.stride_size, fewshot_len, self.stride_size)
            abs_diff = torch.abs(starting_positions - ideal_block_indices.unsqueeze(1))
            # Find the index of the minimum absolute difference for each element in A
            closest_indices = torch.argmin(abs_diff, dim=1)
            # Gather the closest values in B using the indices
            block_indices = list(starting_positions[closest_indices])
        else: 
            block_indices = starting_positions[::self.examples_stride]

        # make a fancy attention mask!
        temp_mask = torch.zeros(fewshot_len, fewshot_len, dtype=torch.bool)
        #temp_mask =  torch.ones(input_len, input_len, dtype=torch.bool).tril(diagonal=0).to(self.model.device) * (attention_mask.unsqueeze(1))
        block_indices.append(fewshot_len) # mark this as the last index in the list
                                
        for i, boundary in enumerate(block_indices):
            mask = torch.tril(torch.ones(fewshot_len, fewshot_len, dtype=torch.bool))

            # select area of the attention
            mask[:, :boundary] = False
            if i + 1 == len(block_indices):
                pass # don't need to zero out to the right of this block
            else:
                mask[:, block_indices[i+1]:] = False
            
            # make that attention ignore things *prior* to this block
            # (don't need to ignore things *after* this block since we already do that with causal attention)
            mask[:boundary,:] = False # block
            if i + 1 == len(block_indices):
                pass # don't need to worry about what's after this block
            else:
                mask[block_indices[i+1]:,:] = False

            temp_mask[mask] = 1
        return temp_mask

    def get_predicted_labels_retrieval(self, restrictive_logit_preprocessor):       
        predicted_labels = []
        
        index = create_retriever(self.train_df)
                    
        for q in tqdm(self.test_df[PROMPTS]):
            context = retrieve_context(train_df=self.train_df, index=index, curr_example=q, \
                n_examples=self.n_shots_per_window, split_text = TEXT_BETWEEN_SHOTS, shuffle_seed=self.window_shuffle_seed)
            fewshot_examples = self.tokenizer(context, add_special_tokens=False, return_tensors='pt')
            fewshot_len = fewshot_examples['input_ids'].shape[-1]

            if self.stride_size != -1 or self.examples_stride != -1:
                temp_mask = self.make_block_mask(fewshot_examples, context)
                
            assert q == q.rstrip(), "prompt ends with a space!"
            encoded_task_text = self.tokenizer(TEXT_BETWEEN_SHOTS+q, add_special_tokens=False, return_tensors='pt')
            encoded_task_text['input_ids'] = encoded_task_text['input_ids'][:, 1:] #cut off '' at the beginning (token 29871)
            encoded_task_text['attention_mask'] = encoded_task_text['attention_mask'][:, 1:] #cut off '' at the beginning (token 29871)

            
            encoded_inputs = torch.cat((fewshot_examples['input_ids'], encoded_task_text['input_ids']), dim=-1).to(self.model.device)
            attention_mask = torch.cat((fewshot_examples['attention_mask'], encoded_task_text['attention_mask']), dim=-1).to(self.model.device)
            custom_attention_mask = None
            input_len = encoded_inputs.shape[-1]
   
            if self.stride_size != -1 or self.examples_stride != -1:
                full_attention_region = encoded_task_text['input_ids'].shape[-1]
                
                # ensure that the full attention region really is full attention (after doing the block attention elsewhere)

                causal_ones = torch.ones(full_attention_region, fewshot_len, dtype=torch.bool)

                this_example_mask=torch.cat((temp_mask, causal_ones), axis=0)


                causal_zeros = torch.zeros(fewshot_len+full_attention_region,full_attention_region, dtype=torch.bool)
                this_example_mask=torch.cat((this_example_mask, causal_zeros), axis=1)
                
                
                mask = torch.tril(torch.ones(fewshot_len+full_attention_region,fewshot_len+full_attention_region, dtype=torch.bool))

                mask[:,:-full_attention_region] = False
                this_example_mask[mask] = 1
                
                custom_attention_mask = (this_example_mask.bool()).unsqueeze(0).unsqueeze(1).to(self.model.device)

            
            kwargs = dict(input_ids=encoded_inputs,
                          custom_attention_mask=custom_attention_mask,
                          attention_mask=attention_mask,
                          #position_ids=torch.arange(0,input_len).unsqueeze(0).to(self.model.device),
                          do_sample=False,
                          num_beams=1,
                          pad_token_id=self.tokenizer.eos_token_id,
                          max_new_tokens=self.max_n_tokens)
            if restrictive_logit_preprocessor is not None:
                restrictive_logit_preprocessor.update_new_prompt_length_to_skip(encoded_inputs.shape[1])
                kwargs['logits_processor'] = [restrictive_logit_preprocessor]

            with torch.no_grad():
                res = self.model.generate(**kwargs)[0]
                res = res[:-1] if res[-1] == self.tokenizer.eos_token_id else res
                predicted_label = self.tokenizer.decode(res[encoded_inputs.shape[1]:])
            predicted_labels.append(predicted_label.lstrip().strip(STOP_SEQUENCE))
            if restrictive_logit_preprocessor is not None:
                assert set(predicted_labels).issubset(self.possible_labels)
            else:
                # clip prediction
                predicted_labels[-1] = predicted_labels[-1].split('\n')[0].split('==')[0].split('source:')[0].rstrip() # we assume batch size of 1 anyway...  hardcoded for smcalflow at the moment but can change the split to use the x_prefix and the examplifier delimeters to be more general if we need
        return predicted_labels
    
    def get_predicted_labels_nonpcw(self, context: str, restrictive_logit_preprocessor):       
        predicted_labels = []
                
        fewshot_examples = self.tokenizer(context, add_special_tokens=False, return_tensors='pt')
        fewshot_len = fewshot_examples['input_ids'].shape[-1]

        if self.stride_size != -1 or self.examples_stride != -1:
            temp_mask = self.make_block_mask(fewshot_examples, context)
                    
        for q in tqdm(self.test_df[PROMPTS]):
            assert q == q.rstrip(), "prompt ends with a space!"
            encoded_task_text = self.tokenizer(TEXT_BETWEEN_SHOTS+q, add_special_tokens=False, return_tensors='pt')
            encoded_task_text['input_ids'] = encoded_task_text['input_ids'][:, 1:] #cut off '' at the beginning (token 29871)
            encoded_task_text['attention_mask'] = encoded_task_text['attention_mask'][:, 1:] #cut off '' at the beginning (token 29871)

            
            encoded_inputs = torch.cat((fewshot_examples['input_ids'], encoded_task_text['input_ids']), dim=-1).to(self.model.device)
            attention_mask = torch.cat((fewshot_examples['attention_mask'], encoded_task_text['attention_mask']), dim=-1).to(self.model.device)
            custom_attention_mask = None
            input_len = encoded_inputs.shape[-1]
   
            if self.stride_size != -1 or self.examples_stride != -1:
                full_attention_region = encoded_task_text['input_ids'].shape[-1]
                
                # ensure that the full attention region really is full attention (after doing the block attention elsewhere)

                causal_ones = torch.ones(full_attention_region, fewshot_len, dtype=torch.bool)

                this_example_mask=torch.cat((temp_mask, causal_ones), axis=0)


                causal_zeros = torch.zeros(fewshot_len+full_attention_region,full_attention_region, dtype=torch.bool)
                this_example_mask=torch.cat((this_example_mask, causal_zeros), axis=1)
                
                
                mask = torch.tril(torch.ones(fewshot_len+full_attention_region,fewshot_len+full_attention_region, dtype=torch.bool))

                mask[:,:-full_attention_region] = False
                this_example_mask[mask] = 1
                
                custom_attention_mask = (this_example_mask.bool()).unsqueeze(0).unsqueeze(1).to(self.model.device)

            
            kwargs = dict(input_ids=encoded_inputs,
                          custom_attention_mask=custom_attention_mask,
                          attention_mask=attention_mask,
                          #position_ids=torch.arange(0,input_len).unsqueeze(0).to(self.model.device),
                          do_sample=False,
                          num_beams=1,
                          pad_token_id=self.tokenizer.eos_token_id,
                          max_new_tokens=self.max_n_tokens)
            if restrictive_logit_preprocessor is not None:
                restrictive_logit_preprocessor.update_new_prompt_length_to_skip(encoded_inputs.shape[1])
                kwargs['logits_processor'] = [restrictive_logit_preprocessor]

            with torch.no_grad():
                res = self.model.generate(**kwargs)[0]
                res = res[:-1] if res[-1] == self.tokenizer.eos_token_id else res
                predicted_label = self.tokenizer.decode(res[encoded_inputs.shape[1]:])
            predicted_labels.append(predicted_label.lstrip().strip(STOP_SEQUENCE))
            if restrictive_logit_preprocessor is not None:
                assert set(predicted_labels).issubset(self.possible_labels)
            else:
                # clip prediction
                predicted_labels[-1] = predicted_labels[-1].split('\n')[0].split('==')[0].split('source:')[0].rstrip() # we assume batch size of 1 anyway...  hardcoded for smcalflow at the moment but can change the split to use the x_prefix and the examplifier delimeters to be more general if we need
        return predicted_labels
    
    def get_predicted_labels(self, windows_few_shots: List[str]) -> List[str]:
        windows_cache = self.model.get_contexts_cache(windows_few_shots)
       
        predicted_labels = []
        raw_labels = []
        
        for q in tqdm(self.test_df[PROMPTS]):
            predicted_label = self.predict_label(TEXT_BETWEEN_SHOTS + q, windows_cache)
            predicted_label = predicted_label.strip()
            predicted_labels.append(predicted_label)
            assert set(predicted_labels).issubset(self.possible_labels)
        return predicted_labels


    def predict_label(self, task_text: str, cache: Dict) -> str:
        assert task_text == task_text.rstrip(), "prompt ends with a space!"
        res = self.model.pcw_generate(task_text=task_text,
                                      contexts_cache=cache,
                                      restrictive_logit_preprocessor=self.logit_processor,
                                      temperature=0,
                                      max_new_tokens=self.max_n_tokens)
        
        return res.lstrip().strip(STOP_SEQUENCE)

    def calc_acc(self, predicted_labels: List, prompt: str) -> float:
        predicted_labels = pd.Series(predicted_labels, index=self.test_df.index, name='outputs')
        multieval = False
        
        if 'labels' in self.test_df:
            multieval = True
            true_labels = self.test_df['labels']
        else:
            # normal eval
            true_labels = self.test_df[LABEL_TOKENS]
            
            try:
                true_labels = true_labels.map(self.label_map)
            except:
                pass # already mapped!
        
        save_state = pd.concat([predicted_labels, true_labels], axis=1)
        save_state['true_numeric_labels'] = self.test_df["label"]
        save_state['true_label_present_in_prompt'] = save_state['true_numeric_labels'].isin(self.label_distribution_prompt)

        if multieval:
            # from chatgpt, proceed w caution
            # Function to check if 'output' is in 'labels'
            def check_output_in_labels(row):
                return row['outputs'] in row['labels']

            # Add a new column to check if 'output' is in 'labels'
            save_state['correct'] = save_state.apply(check_output_in_labels, axis=1)
        else:
            save_state['correct'] = save_state['outputs'] == save_state['label_tokens']

        acc = np.mean(save_state['correct'])
        _logger.info(f"accuracy = {np.round(acc, 3)}")

        if self.is_classification:
            save_state['outputs'] = save_state['outputs'].map(self.reverse_label_map)
            # hacky -- get the shortened text labels that correspond to each of the labels
            textual_prompt_labels = list(map(self.label_map, [self.train_df.loc[self.train_df.index[self.train_df['label'] == i][0]]['label_tokens'] for i in self.label_distribution_prompt]))
            save_state['predicted_label_present_in_prompt'] = save_state['outputs'].isin(textual_prompt_labels)

            save_state['prompt_labels'] = str(self.label_distribution_prompt)

        return acc, save_state

    def run_experiment_across_shots(self, n_shots_to_test: List[int], n_runs: int,
                                    prefix: str, suffix: str,
                                    too_long_patience: float = 0.2,
                                    context_window_size: int = 4096):
        accuracies = np.zeros((len(n_shots_to_test), n_runs))
        predictions = [] #np.zeros((len(n_shots_to_test), n_runs))
        for i, n_shots in enumerate(tqdm(n_shots_to_test)):
            predictions_row = []
            _logger.info(f"starting with n = {n_shots}")
            self._set_random_seed(self.base_random_seed + n_shots)
            j = 0
            n_errors = 0
            while j < n_runs:
                few_shots_idx = self.sample_n_shots(n_shots)
                self.label_distribution_prompt = dict(Counter(self.train_df.loc[few_shots_idx, "label"]))
                selected = self.train_df.loc[few_shots_idx]
                if self.sort_by_label:
                    selected = selected.sort_values("label")
                    
                few_shots_prompts = list(selected[PROMPTS])
                if self.window_shuffle_seed:
                    prev_state = random.getstate()
                    random.seed(self.window_shuffle_seed)
                    random.shuffle(few_shots_prompts)
                    random.setstate(prev_state)
                    
                windows_few_shots = self.build_windows_few_shots_text(few_shots_prompts, self.n_shots_per_window)
                longest_window_n_tokens = max(n_tokens_in_prompt(self.tokenizer, window)
                                              for window in windows_few_shots)
                n_tokens_between_shots = n_tokens_in_prompt(self.tokenizer, TEXT_BETWEEN_SHOTS)

                # check if too long
                if ((longest_window_n_tokens + n_tokens_between_shots + self.test_df[N_TOKENS].max()
                        + self.max_n_tokens) > context_window_size):
                    _logger.warning("Drawn training shots were too long, trying again")
                    n_errors += 1
                    assert n_errors <= too_long_patience * n_runs, "too many long inputs were drawn!"
                    continue
                accuracies[i, j], this_prediction = self.get_few_shots_acc(windows_few_shots)
                this_prediction['prompt_example_indices'] = str(list(few_shots_idx))
                predictions_row.append(this_prediction) 
                j += 1
            predictions.append(predictions_row)
        return accuracies, predictions

    def sample_n_shots(self, n_shots: int) -> npt.NDArray[int]:
        if self.times_shuffled >= len(self.random_orders):
            self.times_shuffled = 0
            self.random_orders = [np.random.permutation(list(self.train_df.index)) for i in range(20)]
            
        few_shots_df = self.train_df.loc[self.random_orders[self.times_shuffled][:n_shots]]
        if self.sync_shuffle:
            self.test_df = self.full_test_df.loc[self.random_orders[self.times_shuffled][:self.subsample_test_set]]
        
        assert few_shots_df.index.is_unique, "few shots samples were not unique!"
        window_size = self.n_shots_per_window or n_shots
        n_windows = int(len(few_shots_df) / window_size)
        self.times_shuffled += 1

        if not self.n_shots_per_window or n_windows == 1:
            return few_shots_df.index

        return self.balance_windows_sizes(n_windows, few_shots_df)

    def balance_windows_sizes(self, n_windows: int, few_shots_df: pd.DataFrame) -> npt.NDArray[int]:
        few_shots_df.sort_values(by=N_TOKENS, inplace=True, ascending=False)
        shape = (self.n_shots_per_window, n_windows)
        indexes = np.array(few_shots_df.index).reshape(shape)
        sizes = few_shots_df.loc[indexes.flatten()].n_tokens.values.reshape(indexes.shape)
        for i in range(1, self.n_shots_per_window):
            order = np.argsort((np.sum(sizes[:i, :], axis=0)))
            sizes[i, :] = sizes[i, order]
            indexes[i, :] = indexes[i, order]
        indexes = indexes.T.flatten()
        return indexes

    @staticmethod
    def build_windows_few_shots_text(few_shots_prompts: List, window_size: int) -> List[str]:
        if window_size is None:
            window_size = len(few_shots_prompts)
        return [TEXT_BETWEEN_SHOTS.join(few_shots_prompts[i: i + window_size]) for i in
                range(0, len(few_shots_prompts), window_size)]
