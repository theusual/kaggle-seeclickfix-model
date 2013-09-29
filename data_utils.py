'''
Generic utility functions useful in data-mining projects
'''
__author__ = 'Bryan Gregory'
__email__ = 'bryan.gregory1@gmail.com'
__date__ = '09-06-2013'

import gc

def data_garbage_collection(dfTrn,dfTest,dfAll):
    # Clean up unused frames:
    dfTrn[0] = '';dfTrn[2] = '';
    dfTest[0] = '';dfTest[2] = '';
    dfAll[1] = ''

    #garbage collection on memory
    gc.collect();
    return dfTrn,dfTest,dfAll


##for split data, use split_data.py

##for turning predictions file to submission csv file, use p2sub.py