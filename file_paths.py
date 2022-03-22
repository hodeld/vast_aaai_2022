from os.path import dirname, abspath, join


file_parent = dirname(abspath(__file__))
misc_data = 'misc_data'
lexica = 'English Lexica'

p_new_contexts = join(file_parent, misc_data, 'new_contexts')
p_cwe_dictionaries = join(file_parent, misc_data, 'cwe_dictionaries')
p_cola_test = join(file_parent, misc_data, 'cola_test')

p_belleza_lexicon = join(file_parent, lexica,  'Bellezza_Lexicon.csv')
p_anew_lexicon = join(file_parent, lexica, 'ANEW.csv')
p_warriner_lexicon = join(file_parent, lexica, 'Warriner_Lexicon.csv')

