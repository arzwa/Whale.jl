#!/usr/bin/env python3
# coding: utf-8
# author: Arthur Zwaenepoel
import sys
import textwrap
DESCRIPTION = """Use this script to filter a bunch of CCD files base on a Poisson outlier criterion. A gene family is deemed an outlier when the square root of the number of clades in its CCD exceeds `n x median[square root(number of clades for gene families with the same number of taxa)]`. Default `n = 2`. See e.g. https://stats.stackexchange.com/questions/56402/detecting-outliers-in-count-data
"""
if not 2 <= len(sys.argv) <= 3:
    print("Usage: {0} <ccd_data.csv> [n]".format(sys.argv[0]))
    wrapper = textwrap.TextWrapper(width=80)
    desc = wrapper.wrap(text=DESCRIPTION)
    for l in desc:
        print("  | {}".format(l))
    exit()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

df = pd.read_csv(sys.argv[1], index_col=0)
if len(sys.argv) == 3:
    n = float(sys.argv[2])
else:
    n = 2

gdf = df.groupby("ntaxa")["nclades"].aggregate(list)
thresh = gdf.apply(lambda x: (0.5*n*np.median(2*np.sqrt(x)))**2)   # n*median(Y), Y = 2sqrt(X)
thresh.name = "thresh"
df = df.join(thresh, on="ntaxa")
df["outlier"] = df["nclades"] > df["thresh"]

def colfun(x):
    if x: return "r"
    else: return "k"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6,3))
ax1.scatter(df["ntaxa"], df["nclades"], s=8, alpha=0.5, c=df["outlier"].apply(colfun))
ax1.set_yscale("log")
ax1.set_ylabel("# clades")
ax1.set_xlabel("# taxa");
x = df[df["outlier"] == False]
y = df[df["outlier"] == True]
bins, edges = np.histogram(np.log10(df["nclades"]/df["ntaxa"]),bins=25)
ax2.hist(np.log10(x["nclades"]/x["ntaxa"]), edges, alpha=0.5, color="k", rwidth=0.8)
ax2.hist(np.log10(y["nclades"]/y["ntaxa"]), edges, alpha=0.5, color="r", rwidth=0.8)
ax2.set_yscale("log")
ax2.set_ylabel("# CCDs")
ax2.set_xlabel("log(# clades/# taxa)");
fig.tight_layout()
fig.savefig("./aleoutliers.pdf")

# print out non-outliers
print("\n".join(df[df["outlier"] == False].index))
