import random

from nltk.tag import pos_tag
import numpy as np
import pandas as pd
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
from string import punctuation
from os import path, listdir
import pickle
import copy
from vast_aaai_2022.file_paths import p_cola_test, p_cwe_dictionaries, p_new_contexts, p_cola_in_domain_dev, \
    p_term_contexts_dic, p_ws353_csv
from vast_aaai_2022.helper_functions import get_embeddings
from nltk import word_tokenize

from vast_aaai_2022.pushshift_data import load_context_dict
from vast_aaai_2022.terms import term_list_weat, anew_terms, warriner_valence_dict, anew_valence, \
    bellezza_terms, \
    bellezza_valence, pleasant_weat, unpleasant_weat, neutral_weat, tokenization_analysis_terms, term_list_weat


def set_embeddings(emb_d, t, c, model, current_tokenizer, tens_type):
    try:
        emb = get_embeddings(t, c, model, current_tokenizer, tensor_type=tens_type)
    except ValueError:
        return
    emb_d[t] = emb


def set_aligned(t_list):
    """
    for different term lists
    term_class_dict and aligned_context_dict include all terms
    """
    for idx, term in enumerate(t_list):
        context = aligned_context_dict[term_class_dict[term]].replace('WORD', term)
        set_embeddings(embedding_dict, term, context, CURRENT_MODEL, CURRENT_TOKENIZER, TENSOR_TYPE)

    with open(path.join(DUMP_PATH, f'{WRITE_MODEL}_{SETTING}.pkl'), 'wb') as pkl_writer:
        pickle.dump(embedding_dict, pkl_writer)

    print(f'{SETTING} done')


def get_existing(t_list=None, fname=None):
    if t_list is None:
        t_list = term_list
    if fname is None:
        fname = f'{WRITE_MODEL}_{SETTING}.pkl'

    with open(path.join(DUMP_PATH, fname), 'rb') as rf:
        emb_d = pickle.load(rf)

    missing_i = [m for m in t_list if m not in emb_d.keys()]
    return emb_d, missing_i


#Model
MODEL_ID_GPT2 = 'gpt2'
MODEL_GPT2 = TFGPT2LMHeadModel.from_pretrained(MODEL_ID_GPT2, output_hidden_states = True, output_attentions = False)
MODEL_TOKENIZER_GPT2 = GPT2Tokenizer.from_pretrained(MODEL_ID_GPT2)

CURRENT_MODEL = MODEL_GPT2
CURRENT_TOKENIZER = MODEL_TOKENIZER_GPT2
WRITE_MODEL = 'gpt2'
TEMPLATE = 'This is WORD'
DUMP_PATH = p_cwe_dictionaries
TENSOR_TYPE = 'tf'

DO_BLEACHED = False
DO_RANDOM = False
DO_TRUE_RANDOM = True
DO_ALIGNED = False
DO_MISALIGNED = False
DO_COLA_TEST_EMBEDDINGS = False
DO_CREATE_TOKENIZE_DICT = False
DO_WS353 = False
DO_EXTEND_EXISTING = True

term_list = term_list_weat


with open(path.join(p_cwe_dictionaries, 'random_context_dictionary.pkl'), 'rb') as pkl_reader:
    context_dict = pickle.load(pkl_reader)

missing = [m for m in term_list if m not in context_dict.keys()]
print('missing: ', len(missing))

#Set valence for aligned contexts
if DO_ALIGNED or DO_MISALIGNED:

    term_valence_dict = copy.deepcopy(warriner_valence_dict)

    for idx, term in enumerate(anew_terms):
        if term not in term_valence_dict:
            term_valence_dict[term] = anew_valence[idx]

    #Rescale Bellezza for consistency with other lexica from 1-5 to 1-9
    for idx, term in enumerate(bellezza_terms):
        if term not in term_valence_dict:
            term_valence_dict[term] = ((bellezza_valence[idx] - 1) * 2) + 1

    for term in pleasant_weat:
        if term not in term_valence_dict:
            term_valence_dict[term] = 8.0

    for term in unpleasant_weat:
        if term not in term_valence_dict:
            term_valence_dict[term] = 2.0

    for term in neutral_weat:
        if term not in term_valence_dict:
            term_valence_dict[term] = 5.0

    term_class_dict = {}

    # groups valence of terms
    for term, valence in term_valence_dict.items():
        if valence <= 2.5:
            term_class_dict[term] = 0
        elif valence <= 4.0:
            term_class_dict[term] = 1
        elif valence <= 6.0:
            term_class_dict[term] = 2
        elif valence <= 7.5:
            term_class_dict[term] = 3
        else:
            term_class_dict[term] = 4

    # 4 highest valence -> context aligned with word
    aligned_context_dict = {0: 'It is very unpleasant to think about WORD',
        1: 'It is unpleasant to think about WORD',
        2: 'It is neither pleasant nor unpleasant to think about WORD',
        3: 'It is pleasant to think about WORD',
        4: 'It is very pleasant to think about WORD',}

    misaligned_context_dict = {4: 'It is very unpleasant to think about WORD',
        3: 'It is unpleasant to think about WORD',
        2: 'It is neither pleasant nor unpleasant to think about WORD',
        1: 'It is pleasant to think about WORD',
        0: 'It is very pleasant to think about WORD',}

#Collect embeddings and write to a dictionary -> because of missing in random always new dict

if DO_BLEACHED:
    SETTING = 'bleached'
    embedding_dict = {}

    for idx, term in enumerate(term_list):
        context = TEMPLATE.replace('WORD', term)  # writes "this is WORD"
        set_embeddings(embedding_dict, term, context, CURRENT_MODEL, CURRENT_TOKENIZER,  TENSOR_TYPE)

    with open(path.join(DUMP_PATH, f'{WRITE_MODEL}_{SETTING}.pkl'), 'wb') as pkl_writer:
        pickle.dump(embedding_dict, pkl_writer)

    print(f'{SETTING} done')

if DO_RANDOM:
    SETTING = 'random'
    if DO_EXTEND_EXISTING:
        embedding_dict, term_list_i = get_existing()
    else:
        term_list_i = term_list
        embedding_dict = {}

    for idx, term in enumerate(term_list_i):
        if context_dict.get(term, None) is None:
            print(f'missing term in context_dict: {term}')
            continue
        context = context_dict[term]
        set_embeddings(embedding_dict, term, context, CURRENT_MODEL, CURRENT_TOKENIZER, TENSOR_TYPE)

    with open(path.join(DUMP_PATH, f'{WRITE_MODEL}_{SETTING}.pkl'), 'wb') as pkl_writer:
        pickle.dump(embedding_dict, pkl_writer)

    print(f'{SETTING} done')

if DO_TRUE_RANDOM:
    SETTING = 'true_random'
    if DO_EXTEND_EXISTING:
        embedding_dict, term_list_i = get_existing()
    else:
        term_list_i = term_list
        embedding_dict = {}

    len_li = len(term_list_i)
    for term in term_list_i:
        if context_dict.get(term, None) is None:
            print(f'missing term in context_dict: {term}')
            continue
        random_idx = random.randint(0, len_li)
        random_term = term_list_i[random_idx]
        random_context = context_dict[random_term]
        context = random_context.replace(random_term, term)
        set_embeddings(embedding_dict, term, context, CURRENT_MODEL, CURRENT_TOKENIZER, TENSOR_TYPE)

    with open(path.join(DUMP_PATH, f'{WRITE_MODEL}_{SETTING}.pkl'), 'wb') as pkl_writer:
        pickle.dump(embedding_dict, pkl_writer)

    print(f'{SETTING} done')

if DO_ALIGNED:
    SETTING = 'aligned'
    embedding_dict = {}
    set_aligned(term_list, suffix='')

if DO_MISALIGNED:
    SETTING = 'misaligned'
    embedding_dict = {}

    for idx, term in enumerate(term_list):
        context = misaligned_context_dict[term_class_dict[term]].replace('WORD', term)
        set_embeddings(embedding_dict, term, context, CURRENT_MODEL, CURRENT_TOKENIZER, TENSOR_TYPE)

    with open(path.join(DUMP_PATH, f'{WRITE_MODEL}_{SETTING}.pkl'), 'wb') as pkl_writer:
        pickle.dump(embedding_dict, pkl_writer)

    print(f'{SETTING} done')

if DO_COLA_TEST_EMBEDDINGS:
    #  Get CoLA Test Embeddings
    k = pd.read_csv(p_cola_in_domain_dev, sep='\t', header=None)

    ids = k.index.to_list()
    labels = k[1].to_list()
    label_dict = {ids[idx]:labels[idx] for idx in range(len(ids))}

    sentences = k[3].to_list()
    sentence_dict = {ids[idx]:sentences[idx] for idx in range(len(ids))}

    sentences = [i.strip() for i in sentences]
    new_sentences = [i.rstrip(punctuation) for i in sentences]
    new_sentence_dict = {ids[idx]:new_sentences[idx] for idx in range(len(ids))}

    actual_last_word = [i.rsplit(' ',1)[1] for i in new_sentences]
    trunced = [i.rsplit(' ',1)[0] for i in new_sentences]

    last_dict = {}
    trunc_dict = {}
    gpt2_predictions = {}
    gpt2_pos = {}
    no_punct_dict = {}

    for idx in range(len(ids)):
        sentence = trunced[idx]
        encoded = MODEL_TOKENIZER_GPT2.encode(sentence,return_tensors='tf')
        output = MODEL_GPT2(encoded)
        last_hs = np.array(output[-1][12][0][-1])
        trunc_dict[idx] = last_hs

        pred = np.argmax(np.squeeze(output[0])[-1])
        next_word = MODEL_TOKENIZER_GPT2.decode([pred])
        gpt2_predictions[idx] = next_word

        new_ = sentence + next_word
        pos = pos_tag(word_tokenize(new_))[-1]
        gpt2_pos[idx] = pos

    with open(path.join(p_cola_test, 'trunc_vectors_val.pkl'),'wb') as pkl_writer:
        pickle.dump(trunc_dict,pkl_writer)

    with open(path.join(p_cola_test, 'gpt2_preds_val.pkl'),'wb') as pkl_writer:
        pickle.dump(gpt2_predictions,pkl_writer)

    with open(path.join(p_cola_test, 'gpt2_pred_pos_val.pkl'),'wb') as pkl_writer:
        pickle.dump(gpt2_pos,pkl_writer)

    print('done trunced')

    for idx in range(len(ids)):
        sentence = sentences[idx]
        encoded = MODEL_TOKENIZER_GPT2.encode(sentence,return_tensors='tf')
        output = MODEL_GPT2(encoded)
        last_hs = np.array(output[-1][12][0][-1])
        last_dict[idx] = last_hs

    with open(path.join(p_cola_test, 'last_vectors_val.pkl'),'wb') as pkl_writer:
        pickle.dump(last_dict,pkl_writer)

    print('done last')

    for idx in range(len(ids)):
        sentence = new_sentences[idx]
        encoded = MODEL_TOKENIZER_GPT2.encode(sentence,return_tensors='tf')
        output = MODEL_GPT2(encoded)
        last_hs = np.array(output[-1][12][0][-1])
        no_punct_dict[idx] = last_hs

    with open(path.join(p_cola_test, 'no_punct_vectors_val.pkl'),'wb') as pkl_writer:
        pickle.dump(no_punct_dict,pkl_writer)


if DO_CREATE_TOKENIZE_DICT:  # note: used for vast
    tokenizer = CURRENT_TOKENIZER
    SETTING = 'true_random'
    fname = f'tokenization_dictionary_{WRITE_MODEL}.pkl'
    if DO_EXTEND_EXISTING:
        tokenization_d, term_list_i = get_existing(t_list=tokenization_analysis_terms, fname=fname)
    else:
        term_list_i = tokenization_analysis_terms
        tokenization_d = {}

    for term in term_list_i:
        tokenized_term = tokenizer.encode(term, add_special_tokens=False, add_prefix_space=True)
        tokenization_d[term] = len(tokenized_term)

    with open(path.join(p_cwe_dictionaries, fname), 'wb') as pkl_writer:
        pickle.dump(tokenization_d, pkl_writer)
    print('tokenization_dictionary done')

if DO_WS353:
    embedding_dict = {}
    SETTING = 'bleached'
    ws353 = pd.read_csv(p_ws353_csv, sep=',')
    word_1 = ws353['Word 1'].to_list()
    word_2 = ws353['Word 2'].to_list()
    fname = f'{WRITE_MODEL}_ws353_dict.pkl'
    ws_terms = list(set(word_1 + word_2))
    if DO_EXTEND_EXISTING:
        embedding_dict, term_list_i = get_existing(t_list=ws_terms, fname=fname)
    else:
        term_list_i = ws_terms
        embedding_dict = {}

    for term in term_list_i:
        context = TEMPLATE.replace('WORD', term)  # writes "this is WORD"
        set_embeddings(embedding_dict, term, context, CURRENT_MODEL, CURRENT_TOKENIZER, TENSOR_TYPE)

    with open(path.join(p_cwe_dictionaries, fname), 'wb') as pkl_writer:
        pickle.dump(embedding_dict, pkl_writer)

    print(f'WS353_{SETTING} done')


print('done everything')