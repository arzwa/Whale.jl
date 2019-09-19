#!/usr/bin/env python3
# coding: utf-8
# author: Arthur Zwaenepoel
import sys
import textwrap
DESCRIPTION = """Use this script to filter an OrthoFinder output data frame based on several criteria. The different criteria that can be used are: (0) at least one gene in both clades: for this you have to specify one clade as a comma-separated string of taxa, and the data frame will be filtered such that there is at least one gene both in the specified clade and its complement. (1) Outlier families: this filters out large families based on a Poisson outlier criterion, i.e. exclude families where the square root of the family size exceeds 2 x median(square root of family size). (2) Outlier lineages: This filters out families where at least one of the lineages has a number of genes that exceeds the Poisson outlier criterion. To specify for instance filter 0 and 1 (recommended filters), use '0,1' as last argument
"""
if not 4 <= len(sys.argv) <= 5:
    print("Usage: {} <orthogroups.tsv> <clade1> <out> [filter]".format(sys.argv[0]))
    wrapper = textwrap.TextWrapper(width=80)
    desc = wrapper.wrap(text=DESCRIPTION)
    for l in desc:
        print("  | {}".format(l))
    exit()
import pandas as pd
import numpy as np

if len(sys.argv) == 5:
    filters = [int(x) for x in sys.argv[4].split(",")]
else:
    filters = [0, 1]

def filter_outliers1(df):
    df["Y"] = 2*np.sqrt(df["total"])
    return df[df["Y"] < np.median(df["Y"]) + 3]

def filter_outliers2(df):
    df_ = df
    if "total" in df.columns:
        df_.drop("total", axis=1, inplace=True)
    if "Y" in df.columns:
        df_.drop("Y", axis=1, inplace=True)
    return df_[df_.apply(is_notoutlier, axis=1)]

def is_notoutlier(x):
    y = 2*np.sqrt(x)
    f = lambda y: y <= np.median(y) + 3
    return np.all(f(y))


# read data and get gene counts
df = pd.read_csv(sys.argv[1], sep="\t", index_col=0)
df_ = df.fillna("")
counts = df_.applymap(lambda x: len([y for y in x.split("," ) if y != '']))
counts["total"] = counts.apply(sum, axis=1)

# one in both filter and miimum size filter
if 0 in filters:
    clade1 = sys.argv[2].split(",")
    print("Applying filter 0 (clades, {}): {} -> ".format(clade1, len(counts)), end="")
    if len(clade1) == 1:
        counts["clade1"] = counts[clade1[0]]
    else:
        counts["clade1"] = counts[clade1].apply(sum, axis=1)
    counts = counts[counts["clade1"] > 0]
    counts = counts[counts["total"] - counts["clade1"] > 0]
    counts = counts[counts["total"] > 3]
    print(len(counts))

# Poisson outlier filter (families)
if 1 in filters:
    print("Applying filter 1 (outlier families): {} -> ".format(len(counts)), end="")
    counts = filter_outliers1(counts)
    print(len(counts))

# Poisson outliers (lineages)
if 2 in filters:
    print("Applying filter 2 (outlier lineages): {} -> ".format(len(counts)), end="")
    counts = filter_outliers2(counts)
    print(len(counts))

df.loc[counts.index].to_csv(sys.argv[3], sep="\t")
