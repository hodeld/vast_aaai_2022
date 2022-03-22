from os.path import dirname, abspath, join


file_parent = dirname(abspath(__file__))
misc_data = 'misc_data'
lexica = 'English Lexica'
cola = join('cola_public', 'raw')  # downloaded from https://nyu-mll.github.io/CoLA/ | were raw in original file

p_new_contexts = join(file_parent, misc_data, 'new_contexts')
p_cwe_dictionaries = join(file_parent, misc_data, 'cwe_dictionaries')
p_cola_test = join(file_parent, misc_data, 'cola_test')

p_term_contexts_dic = join(p_new_contexts, 'term_contexts_dictionary.pkl')
p_pushshift_missing = join(p_new_contexts, 'missing.pkl')

p_belleza_lexicon = join(file_parent, lexica,  'Bellezza_Lexicon.csv')
p_anew_lexicon = join(file_parent, lexica, 'ANEW.csv')
p_warriner_lexicon = join(file_parent, lexica, 'Warriner_Lexicon.csv')

p_cola_in_domain_dev = join(file_parent, misc_data, cola, 'in_domain_dev.tsv')
p_cola_in_domain_train = join(file_parent, misc_data, cola, 'in_domain_train.tsv')
p_cola_out_domain_dev = join(file_parent, misc_data, cola, 'out_of_domain_dev.tsv')

p_reddit_comments = join(file_parent, misc_data, 'reddit_comments')
p_reddit_comments_file = join(p_reddit_comments, 'RC_2005-12')
p_reddit_comments_missing = join(p_reddit_comments, 'missing.pkl')
