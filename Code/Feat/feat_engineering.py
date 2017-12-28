# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 13:35:17 2017

@author: maoli
"""

import pandas as pd
import numpy as np



def count_single_col(dataset,var_list,keep_list = []):
    for var in var_list:
        if not var in dataset.columns:
            continue
        dataset['cnt_'+var] = 0
        keep_list.append('cnt_'+var)
        groups = dataset.groupby([var])
        for name,group in groups:
            count = group[var].count()
            dataset['cnt_'+var].ix[group.index] = count
    return dataset[keep_list]




def count_by_other_col(dataset,var_dict,keep_list = []):
    for tool,pro in var_dict.items():
        try:
            groups = dataset.groupby(['TOOL_'+tool[-3:]])
        except:
            continue
        for var in pro:
            if not var in dataset.columns:
                continue
            dataset['cnt_'+tool+'_'+var] = 0
            keep_list.append('cnt_'+tool+'_'+var)
            for name,group in groups:
                grps = group.groupby([var])
                for name2,grp in grps:
                    dataset['cnt_'+tool+'_'+var].ix[grp.index] = float(len(grp))/float(len(group))
    return dataset[keep_list]



def pcent_single_col(dataset,var_list,keep_list = []):
    for var in var_list:
        if not var in dataset.columns:
            continue
        if 'TOOL' not in var:
            dataset['pcent_'+var] = dataset[var].rank(method='max')/float(len(dataset))
            keep_list.append('pcent_'+var)
    return dataset[keep_list]


def pcent_by_other_col(dataset,var_dict,keep_list = []):
    for tool,pro in var_dict.items():
        try:
            groups = dataset.groupby(['TOOL_'+tool[-3:]])
        except:
            continue
        for var in pro:
            if not var in dataset.columns:
                continue
            dataset['pcent_'+tool+'_'+var] = 0
            keep_list.append('pcent_'+tool+'_'+var)
            for name,group in groups:
                dataset['pcent_'+tool+'_'+var].ix[group.index] = group[var].rank(method='max')/float(len(group))
    return dataset[keep_list]


def var_gap(dataset,var_list,keep_list = [],criteria=):


























