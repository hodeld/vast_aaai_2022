import json
import pickle
import time
from os import path, listdir
from datetime import datetime

import requests
from nltk import word_tokenize

from vast_aaai_2022.file_paths import p_reddit_comments_file, p_reddit_comments_missing, \
     p_new_contexts, p_reddit_comments, p_cwe_dictionaries
from vast_aaai_2022.terms import term_list_weat

FROM_PICKLE_FILE = False


def load_context_dict():
    context_dict = {}
    dir_ = p_new_contexts
    dir_list = list(listdir(dir_))
    for f in dir_list:
        if 'missing' in f or '.pkl' not in f:
            continue
        print(f)
        with open(path.join(dir_, f), 'rb') as r:
            context_d_i = pickle.load(r)
            context_dict.update(context_d_i)

    return context_dict


def download_comments():
    def check_tokenize_context():
        try:
            tokenized_context = word_tokenize(context)  # uses nltk_data -> downloaded to env/share/nltk_data
            tokenized_term = word_tokenize(term)
        except:
            print('word_tokenize did not work -> check nltk_data')
            return
        # pos = tokenized_context.index(term[0]) takes 1st letter of term instead of whole term -> changed to: tokenized_term[0]
        if tokenized_term[0] not in tokenized_context:
            print('tokenized term not in context', term)
            return
        pos = tokenized_context.index(tokenized_term[0])
        if pos == 0:  # if at beginning: cannot be embedded with get_embeddings
            print('term at beginning', term)
            return
        start = max(0, pos - 10)
        end = min(len(tokenized_context), pos + 10)  # + 11 to make it symmetric
        context_dict[term] = ' '.join(tokenized_context[start:end])
        missing.remove(term)
        return True

    p_main = p_new_contexts
    base_url = 'https://api.pushshift.io/reddit/search/comment/?q={query}&fields=body&size={size}'

    term_list = term_list_weat
    context_dict = load_context_dict()

    missing = [m for m in term_list if m not in context_dict.keys()]

    orig_missing_len = len(missing)
    size_start = 10
    for i in range(5):
        for term in iter(missing):
            url = base_url.format(query=term, size=(i+1)*size_start)
            request = requests.get(url)
            if request.ok is False:
                if request.status_code == 429:
                    time.sleep(1)
                print('term with bad request', term, request.status_code)
                continue
            json_response = request.json()
            for d in json_response['data']:
                context = d['body']
                if check_tokenize_context():
                    break

    if FROM_PICKLE_FILE:
        with open(p_reddit_comments_file, 'r') as f:
            for submissionLine in f:
                context = json.loads(submissionLine)['body']
                if type(context) != str:
                    continue
                for term in missing:
                    if term in context:
                        print(len(missing))
                        check_tokenize_context()
                if not missing:
                    break

    if orig_missing_len > len(missing):
        ending = datetime.now().strftime('-%y%m%d-%H-%M') + '.pkl'
        p_context_dic = path.join(p_new_contexts, 'term_contexts_dictionary' + ending)

        with open(p_context_dic, 'wb') as pkl_writer:
            pickle.dump(context_dict, pkl_writer)

        with open(path.join(p_cwe_dictionaries, 'random_context_dictionary_new.pkl'), 'wb') as pkl_writer:
            pickle.dump(context_dict, pkl_writer)


if __name__ == "__main__":
    download_comments()