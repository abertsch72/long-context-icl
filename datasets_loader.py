import logging
from abc import ABC
from typing import Dict, Optional
import re

import pandas as pd
import json
from datasets import load_dataset

from constants import PROMPTS


UTTERANCE_PREFIX = 'utterance: '

INTENT_PREFIX = 'intent: '

LABEL_TOKENS = 'label_tokens'

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')


class ClassificationDatasetAccess(ABC):
    name: str
    dataset: Optional[str] = None
    subset: Optional[str] = None
    orig_x_column: str = 'text'
    orig_y_label: str='label'
    x_column: str = 'text'
    y_label: str = 'label'
    x_prefix: str = "Review: "
    y_prefix: str = "Sentiment: "
    label_mapping: Optional[Dict] = None
    map_labels: bool = True
    token: str = None
    test_from_train: bool = False
    repeat: bool = False
    local: bool = False
    seed: int = None
    half: bool = False

    def __init__(self, token=None, seed=None):
        super().__init__()
        self.token=token
        if seed is not None:
            self.seed = seed 
            self.half=True
            
        if self.dataset is None:
            self.dataset = self.name
        train_dataset, test_dataset = self._load_dataset(self.test_from_train)
        if self.orig_x_column != self.x_column:
            train_dataset = train_dataset.rename_column(self.orig_x_column, self.x_column)
            test_dataset = test_dataset.rename_column(self.orig_x_column, self.x_column)

        if self.orig_y_label != self.y_label:
            train_dataset = train_dataset.rename_column(self.orig_y_label, self.y_label)
            test_dataset= test_dataset.rename_column(self.orig_y_label, self.y_label)
        
        train_df = train_dataset.to_pandas()
        test_df = test_dataset.to_pandas()
        _logger.info(f"loaded {len(train_df)} training samples & {len(test_df)} test samples")

        if self.map_labels:
            hf_default_labels = train_dataset.features[self.y_label]
            default_label_mapping = dict(enumerate(hf_default_labels.names)) if hasattr(
                train_dataset.features[self.y_label], 'names') else \
                    {x:x for x in set(train_dataset[self.y_label])}  # modification for clinic150
            if self.half:
                default_label_mapping = {label:default_label_mapping[label] for label in default_label_mapping.keys() if label in set(train_df['label'])}
            self._initialize_label_mapping(default_label_mapping)

        self.train_df = self.apply_format(train_df, repeat=self.repeat)
        self.test_df = self.apply_format(test_df, test=True)
        

        
    def _initialize_label_mapping(self, default_label_mapping):
        if self.label_mapping:
            _logger.info("overriding default label mapping")
            if default_label_mapping:
                _logger.info([f"{default_label_mapping[k]} -> "
                              f"{self.label_mapping[k]}" for k in self.label_mapping.keys()])
        else:
            _logger.info(f"using default label mapping: {default_label_mapping}")
            self.label_mapping = default_label_mapping

    def _load_dataset(self, test_from_train=False, prefer_val=False):
        if self.local:
            from datasets import load_from_disk
            dataset = load_from_disk(self.dataset)
        else:
            if self.subset is not None:
                dataset = load_dataset(self.dataset, self.subset, use_auth_token=self.token)
            else:  
                dataset = load_dataset(self.dataset, use_auth_token=self.token)
        if 'validation' in dataset and prefer_val:
            return dataset['train'], dataset['validation']
        if 'test' not in dataset:
            _logger.info("no test or validation found, splitting train set instead")
            dataset = dataset['train'].train_test_split(seed=42)

        if self.half: # use half as many labels!
            all_labels = set(dataset['train'][self.orig_y_label])
            import random
            prev_state = random.getstate()
            random.seed(self.seed)
            chosen_labels = random.sample(all_labels, len(all_labels) // 2)
            dataset = dataset.filter(lambda example: example[self.orig_y_label] in chosen_labels)
            random.setstate(prev_state)
            if self.label_mapping:
                self.label_mapping = {k:self.label_mapping[k] for k in self.label_mapping.keys() if k in chosen_labels}
        # TODO: shuffle data in a deterministic way!
        dataset['train'] = dataset['train'].shuffle(seed=39)
        if test_from_train:
            _logger.warning("Using the same examples for train and test. Do this only as a sanity check!")
            import copy
            return dataset['train'], copy.deepcopy(dataset['train']) #same examples in train and test
        else:
            return dataset['train'], dataset['test'] #actually use a test set, the normal way

    def generate_x_text(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def generate_y_token_labels(self, df, test):
        if self.map_labels:
            df[LABEL_TOKENS] = df[self.y_label].map(self.label_mapping)
        else:
            df[LABEL_TOKENS] = df[self.y_label]
        return df

    @property
    def labels(self):
        if self.map_labels:
            return self.label_mapping.values()
        elif 'labels' in self.test_df:
            labels_test = self.test_df['labels'].explode()
            labels_train = self.train_df['labels'].explode()
            import numpy as np
            return np.concatenate((labels_train.unique(),(labels_test.unique()))) #multiple labels valid for each answer
        else:
            return self.test_df[LABEL_TOKENS].unique()

    @property
    def train_labels(self):
        return self.train_df[LABEL_TOKENS].unique()


    def apply_format(self, df, test=False, repeat=False):
        df = self.generate_x_text(df)
        df = self.generate_y_token_labels(df, test)
        if test:
            df[PROMPTS] = df.apply(lambda x: f"{self.x_prefix}{x[self.x_column]}\n{self.y_prefix}".rstrip(), axis=1)
        else:
            if repeat:
                df[PROMPTS] = df.apply(lambda x: f"{self.x_prefix}{x[self.x_column]}\n{self.y_prefix}{x[LABEL_TOKENS]}\nRepeat: {self.x_prefix}{x[self.x_column]}\n{self.y_prefix}{x[LABEL_TOKENS]}\n",
                                   axis=1)
            else:
                df[PROMPTS] = df.apply(lambda x: f"{self.x_prefix}{x[self.x_column]}\n{self.y_prefix}{x[LABEL_TOKENS]}",
                                   axis=1)
        return df


class SST5(ClassificationDatasetAccess):
    name = 'sst5'
    dataset = 'SetFit/sst5'
    label_mapping = {0: 'terrible', 1: 'bad', 2: 'okay', 3: 'good', 4: 'great'}


class RTE(ClassificationDatasetAccess):
    name = 'rte'
    dataset = 'super_glue'
    subset = 'rte'
    x_prefix = ''
    y_prefix = 'prediction: '
    label_mapping = {0: 'True', 1: 'False'}

    def generate_x_text(self, df: pd.DataFrame) -> pd.DataFrame:
        df['text'] = df.apply(lambda x: f"premise: {x['premise']}\nhypothesis: {x['hypothesis']}", axis=1)
        return df


class CB(RTE):
    name = 'cb'
    subset = 'cb'
    label_mapping = {0: 'true', 1: 'false', 2: 'neither'}


class SUBJ(ClassificationDatasetAccess):
    name = 'subj'
    dataset = 'SetFit/subj'
    label_mapping = {0: 'objective', 1: 'subjective'}
    x_prefix = 'Input: '
    y_prefix = 'Type: '


class CR(ClassificationDatasetAccess):
    name = 'cr'
    dataset = 'SetFit/CR'
    label_mapping = {0: 'negative', 1: 'positive'}


class AGNEWS(ClassificationDatasetAccess):
    name = 'agnews'
    dataset = 'ag_news'
    label_mapping = {0: 'world', 1: 'sports', 2: 'business', 3: 'technology'}
    x_prefix = 'input: '
    y_prefix = 'type: '


class DBPEDIA(ClassificationDatasetAccess):
    name = 'dbpedia'
    dataset = 'dbpedia_14'
    label_mapping = {0: 'company',
                     1: 'school',
                     2: 'artist',
                     3: 'athlete',
                     4: 'politics',
                     5: 'transportation',
                     6: 'building',
                     7: 'nature',
                     8: 'village',
                     9: 'animal',
                     10: 'plant',
                     11: 'album',
                     12: 'film',
                     13: 'book'}
    x_prefix = 'input: '
    y_prefix = 'type: '

    def generate_x_text(self, df: pd.DataFrame) -> pd.DataFrame:
        df['text'] = df['content']
        return df


class SST2(ClassificationDatasetAccess):
    name = 'sst2'

    def generate_x_text(self, df: pd.DataFrame) -> pd.DataFrame:
        df['text'] = df['sentence']
        return df


class TREC(ClassificationDatasetAccess):
    name = 'trec'
    orig_y_label = 'coarse_label'
    x_prefix = "Question: "
    y_prefix = "Type: "
    label_mapping = {0: "abbreviation", 1: "entity", 2: "description", 3: "human", 4: "location", 5: 'numeric'}


class TRECFINE(ClassificationDatasetAccess):
    name = 'trecfine'
    dataset = 'trec'
    orig_y_label = 'fine_label'
    x_prefix = "Question: "
    y_prefix = "Type: "
    # labels mapping based on: https://aclanthology.org/C16-1116.pdf, https://aclanthology.org/C02-1150.pdf
    label_mapping = {0: 'abbreviation abbreviation',
                     1: 'abbreviation expansion',
                     2: 'entity animal',
                     3: 'entity body',
                     4: 'entity color',
                     5: 'entity creation',
                     6: 'entity currency',
                     7: 'entity disease',
                     8: 'entity event',
                     9: 'entity food',
                     10: 'entity instrument',
                     11: 'entity language',
                     12: 'entity letter',
                     13: 'entity other',
                     14: 'entity plant',
                     15: 'entity product',
                     16: 'entity religion',
                     17: 'entity sport',
                     18: 'entity substance',
                     19: 'entity symbol',
                     20: 'entity technique',
                     21: 'entity term',
                     22: 'entity vehicle',
                     23: 'entity word',
                     24: 'description definition',
                     25: 'description description',
                     26: 'description manner',
                     27: 'description reason',
                     28: 'human group',
                     29: 'human individual',
                     30: 'human title',
                     31: 'human description',
                     32: 'location city',
                     33: 'location country',
                     34: 'location mountain',
                     35: 'location other',
                     36: 'location state',
                     37: 'numeric code',
                     38: 'numeric count',
                     39: 'numeric date',
                     40: 'numeric distance',
                     41: 'numeric money',
                     42: 'numeric order',
                     43: 'numeric other',
                     44: 'numeric period',
                     45: 'numeric percent',
                     46: 'numeric speed',
                     47: 'numeric temperature',
                     48: 'numeric size',
                     49: 'numeric weight'}


class YELP(ClassificationDatasetAccess):
    name = 'yelp'
    dataset = 'yelp_review_full'
    x_prefix = 'review: '
    y_prefix = 'stars: '
    label_mapping = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5'}


class BANKING77(ClassificationDatasetAccess):
    name = 'banking77'
    x_prefix = 'query: '
    y_prefix = INTENT_PREFIX

    def _initialize_label_mapping(self, default_label_mapping):
        default_label_mapping = {k: v.replace('_', ' ') for k, v in default_label_mapping.items()}
        super()._initialize_label_mapping(default_label_mapping)

class BANKING77_SANITY(BANKING77):
    test_from_train = True # keep the test and train datasets the same!

class BANKING77_REPEAT(BANKING77):
    repeat = True #use doubled inputs!
   
class NLU(ClassificationDatasetAccess):
    name = 'nlu'
    dataset = 'nlu_evaluation_data'
    x_prefix = UTTERANCE_PREFIX
    y_prefix = INTENT_PREFIX
    label_mapping = {0: 'alarm query', 1: 'alarm remove', 2: 'alarm set', 3: 'audio volume down',
                     4: 'audio volume mute', 5: 'audio volume other', 6: 'audio volume up', 7: 'calendar query',
                     8: 'calendar remove', 9: 'calendar set', 10: 'cooking query', 11: 'cooking recipe',
                     12: 'datetime convert', 13: 'datetime query', 14: 'email add contact', 15: 'email query',
                     16: 'email query contact', 17: 'email sendemail', 18: 'general affirm', 19: 'general command stop',
                     20: 'general confirm', 21: 'general dont care', 22: 'general explain', 23: 'general greet',
                     24: 'general joke', 25: 'general negate', 26: 'general praise', 27: 'general quirky',
                     28: 'general repeat', 29: 'iot cleaning', 30: 'iot coffee', 31: 'iot hue light change',
                     32: 'iot hue light dim', 33: 'iot hue light off', 34: 'iot hue lighton', 35: 'iot hue light up',
                     36: 'iot wemo off', 37: 'iot wemo on', 38: 'lists create or add', 39: 'lists query',
                     40: 'lists remove', 41: 'music dislikeness', 42: 'music likeness', 43: 'music query',
                     44: 'music settings', 45: 'news query', 46: 'play audiobook', 47: 'play game', 48: 'play music',
                     49: 'play podcasts', 50: 'play radio', 51: 'qa currency', 52: 'qa definition', 53: 'qa factoid',
                     54: 'qa maths', 55: 'qa stock', 56: 'recommendation events', 57: 'recommendation locations',
                     58: 'recommendation movies', 59: 'social post', 60: 'social query', 61: 'takeaway order',
                     62: 'takeaway query', 63: 'transport query', 64: 'transport taxi', 65: 'transport ticket',
                     66: 'transport traffic', 67: 'weather query'}


class NLUSCENARIO(ClassificationDatasetAccess):
    name = 'nluscenario'
    dataset = 'nlu_evaluation_data'
    x_prefix = UTTERANCE_PREFIX
    y_prefix = 'scenario: '
    y_label = 'scenario'
    map_labels = False


class TACRED_TEXT(ClassificationDatasetAccess):
    name = 'tacred-text'
    dataset = 'AmirLayegh/tacred_text_label'
    x_prefix = 'Sentence: '
    y_prefix = 'Relation: '
    map_labels = False

class CLINIC150(BANKING77):
    name = "clinic150"
    dataset = 'clinc_oos'
    orig_x_column="text"
    subset = 'plus'
    orig_y_label = "intent"
    x_prefix = UTTERANCE_PREFIX
    y_prefix = INTENT_PREFIX
    label_mapping = {
        0: 'restaurant reviews',
        1: 'nutrition info',
        2: 'account blocked',
        3: 'oil change how',
        4: 'time',
        5: 'weather',
        6: 'redeem rewards',
        7: 'interest rate',
        8: 'gas type',
        9: 'accept reservations',
        10: 'smart home',
        11: 'user name',
        12: 'report lost card',
        13: 'repeat',
        14: 'whisper mode',
        15: 'what are your hobbies',
        16: 'order',
        17: 'jump start',
        18: 'schedule meeting',
        19: 'meeting schedule',
        20: 'freeze account',
        21: 'what song',
        22: 'meaning of life',
        23: 'restaurant reservation',
        24: 'traffic',
        25: 'make call',
        26: 'text',
        27: 'bill balance',
        28: 'improve credit score',
        29: 'change language',
        30: 'no',
        31: 'measurement conversion',
        32: 'timer',
        33: 'flip coin',
        34: 'do you have pets',
        35: 'balance',
        36: 'tell joke',
        37: 'last maintenance',
        38: 'exchange rate',
        39: 'uber',
        40: 'car rental',
        41: 'credit limit',
        42: 'oos',
        43: 'shopping list',
        44: 'expiration date',
        45: 'routing',
        46: 'meal suggestion',
        47: 'tire change',
        48: 'todo list',
        49: 'card declined',
        50: 'rewards balance',
        51: 'change accent',
        52: 'vaccines',
        53: 'reminder update',
        54: 'food last',
        55: 'change ai name',
        56: 'bill due',
        57: 'who do you work for',
        58: 'share location',
        59: 'international visa',
        60: 'calendar',
        61: 'translate',
        62: 'carry on',
        63: 'book flight',
        64: 'insurance change',
        65: 'todo list update',
        66: 'timezone',
        67: 'cancel reservation',
        68: 'transactions',
        69: 'credit score',
        70: 'report fraud',
        71: 'spending history',
        72: 'directions',
        73: 'spelling',
        74: 'insurance',
        75: 'what is your name',
        76: 'reminder',
        77: 'where are you from',
        78: 'distance',
        79: 'payday',
        80: 'flight status',
        81: 'find phone',
        82: 'greeting',
        83: 'alarm',
        84: 'order status',
        85: 'confirm reservation',
        86: 'cook time',
        87: 'damaged card',
        88: 'reset settings',
        89: 'pin change',
        90: 'replacement card duration',
        91: 'new card',
        92: 'roll dice',
        93: 'income',
        94: 'taxes',
        95: 'date',
        96: 'who made you',
        97: 'pto request',
        98: 'tire pressure',
        99: 'how old are you',
        100: 'rollover 401k',
        101: 'pto request status',
        102: 'how busy',
        103: 'application status',
        104: 'recipe',
        105: 'calendar update',
        106: 'play music',
        107: 'yes',
        108: 'direct deposit',
        109: 'credit limit change',
        110: 'gas',
        111: 'pay bill',
        112: 'ingredients list',
        113: 'lost luggage',
        114: 'goodbye',
        115: 'what can i ask you',
        116: 'book hotel',
        117: 'are you a bot',
        118: 'next song',
        119: 'change speed',
        120: 'plug type',
        121: 'maybe',
        122: 'w2',
        123: 'oil change when',
        124: 'thank you',
        125: 'shopping list update',
        126: 'pto balance',
        127: 'order checks',
        128: 'travel alert',
        129: 'fun fact',
        130: 'sync device',
        131: 'schedule maintenance',
        132: 'apr',
        133: 'transfer',
        134: 'ingredient substitution',
        135: 'calories',
        136: 'current location',
        137: 'international fees',
        138: 'calculator',
        139: 'definition',
        140: 'next holiday',
        141: 'update playlist',
        142: 'mpg',
        143: 'min payment',
        144: 'change user name',
        145: 'restaurant suggestion',
        146: 'travel notification',
        147: 'cancel',
        148: 'pto used',
        149: 'travel suggestion',
        150: 'change volume'
    }

    #TODO: is this cleaner than the workaround I did?
    map_labels = True


def get_loader(dataset_name, token=None, half_seed=None):
    if dataset_name in DATASET_NAMES2LOADERS:
        return DATASET_NAMES2LOADERS[dataset_name](token=token, seed=half_seed)
    if ' ' in dataset_name:
        dataset, subset = dataset_name.split(' ')
        if dataset == 'Mivg/sanity_experiment':
            return Sanity(subset, token=token)
    raise KeyError(f'Unknown dataset name: {dataset_name}')

class Sanity(ClassificationDatasetAccess):
    name = 'Mivg/sanity_experiment'
    dataset = 'Mivg/sanity_experiment'
    x_prefix = ''
    y_prefix = ''

    def __init__(self, subset, token=None):
        match = re.match('seq_(\d+)_kv_(\d+)', subset)
        assert match is not None
        self.subset = subset
        self.token = token
        super().__init__()

class ULTRAFINE_ENTITY(ClassificationDatasetAccess):
    name = 'ultrafine-entity'
    dataset = 'data/finegrained-entitylinked-local'
    x_prefix = 'Sentence: '
    y_prefix = 'Type of marked entity: '
    orig_y_label = 'label1'
    orig_x_column = 'marked_sentence'
    map_labels = False
    local = True
    
    def __init__(self, token=None):
        self.token = token
        super().__init__(token=token)

class CLINIC150_SANITY(CLINIC150):
    test_from_train = True # keep the test and train datasets the same!

class TRECFINE_SANITY(TRECFINE):
    test_from_train = True

class NLU_SANITY(NLU):
    test_from_train = True 
    
TEST_ON_TRAIN_DATASETS = ['banking77-sanity', 'clinic150-sanity', 'trecfine-sanity', 'nlu-sanity']
class SMCalFlowCS(ClassificationDatasetAccess):
    name = 'SMCalFlowCS'
    dataset = 'data/smcalflow_cs.jsonl'
    split_path = 'data/smcalflow_cs_source_domain_with_target_num32_split.json'
    train_split = 'train'
    test_split = 'test'  # we don't want the iid split https://github.com/itayle/diverse-demonstrations/blob/650910b01362831f889ddcd0fa15f5bc3db5297f/diverse_demonstrations/evaluate_NOFT.py#L110C52-L110C55
    orig_x_column = "source"
    orig_y_label = "target"
    x_prefix = 'source: '  # https://github.com/itayle/diverse-demonstrations/blob/650910b01362831f889ddcd0fa15f5bc3db5297f/diverse_demonstrations/evaluate_NOFT.py#L229
    y_prefix = 'target: '
    # between examples, we should have \n according to the diverse_demonstrations repo. each line should be f"{x_prefix} {source}\n{y_prefix} {target}\n"
    # our format is df.apply(lambda x: f"{self.x_prefix}{x[self.x_column]}\n{self.y_prefix}{x[LABEL_TOKENS]}", axis=1)
    map_labels = False # the outputs are long unique generations, so it does not make sense to map them
    test_from_train = False

    def _load_dataset(self, test_from_train=False, prefer_val=False):
        if test_from_train:
            raise NotImplementedError('this is against what this dataset is for')
        dataset = load_dataset('json', data_files=self.dataset)

        with open(self.split_path, 'r') as f:
            split_ids = json.load(f)

        def filter_split(example, split_ids):
            return example['qid'] in split_ids

        # Apply the filter function for each split
        train_dataset = dataset['train'].filter(lambda example: filter_split(example, split_ids[self.train_split]))
        if prefer_val:
            test_dataset = dataset['train'].filter(lambda example: filter_split(example, split_ids['validation']))
        else:
            test_dataset = dataset['train'].filter(lambda example: filter_split(example, split_ids[self.test_split]))

        train_dataset = train_dataset.shuffle(seed=39)
        test_dataset = test_dataset.shuffle(seed=39)

        return train_dataset, test_dataset



DATASET_NAMES2LOADERS = {'sst5': SST5, 'sst2': SST2, 'agnews': AGNEWS, 'dbpedia': DBPEDIA, 'trec': TREC, 'cr': CR,
                         'cb': CB, 'rte': RTE, 'subj': SUBJ, 'yelp': YELP, 'banking77': BANKING77,
                         'nlu': NLU, 'nluscenario': NLUSCENARIO, 'trecfine': TRECFINE,
                         'clinic150': CLINIC150, 'clinic150-sanity': CLINIC150_SANITY, 'banking77-sanity': BANKING77_SANITY,
                         'banking77-repeat': BANKING77_REPEAT, 'ultrafine-entity': ULTRAFINE_ENTITY, 'smcalflow_cs': SMCalFlowCS,
                         'nlu-sanity': NLU_SANITY, 'trecfine-sanity': TRECFINE_SANITY, 'tacred-text': TACRED_TEXT}

if __name__ == '__main__':
    for ds_name, da in DATASET_NAMES2LOADERS.items():
        _logger.info(ds_name)
        _logger.info(da().train_df[PROMPTS].iloc[0])
