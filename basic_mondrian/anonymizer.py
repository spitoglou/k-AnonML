# -*- coding: utf-8 -*-
"""
run basic_mondrian with given parameters
"""
import copy
import sys
import os

from basic_mondrian.mondrian import mondrian

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils.data import reorder_columns, restore_column_order


DATA_SELECT = 'a'
DEFAULT_K = 10


def extend_result(val):
    """
    separated with ',' if it is a list
    """
    if isinstance(val, list):
        return ','.join(val)
    return val


def get_result_one(att_trees, data, k, path, qi_index, SA_index, logger=None):
    "run basic_mondrian for one time, with k=10"
    if logger:
        logger.info(f"[MONDRIAN] Starting k={k}")
        logger.debug(f"[MONDRIAN] Algorithm: Mondrian")
    else:
        print("K=%d" % k)
        print("Mondrian")
    
    result, eval_result = mondrian(att_trees, reorder_columns(
        copy.deepcopy(data), qi_index), k, len(qi_index), SA_index)
    
    if logger:
        logger.info(f"[MONDRIAN] NCP: {eval_result[0]:.2f}%, Time: {eval_result[1]:.2f}s")
    else:
        print("NCP %0.2f" % eval_result[0] + "%")
        print("Running time %0.2f" % eval_result[1] + "seconds")
    
    return restore_column_order(result, qi_index)
