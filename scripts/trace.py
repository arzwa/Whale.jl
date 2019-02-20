#!/usr/bin/python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import seaborn as sns
import sys


"""
Trace plots
"""
def plot_trace(df, burn_in=1000, ncol=2, figsize=None, discard=False):
    from math import ceil
    nrow = ceil(len(df.columns)/ncol)
    if not figsize:
        figsize=(12, 4*nrow)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    row, col = 0, 0
    
    if discard:
        df_ = df.loc[burn_in:]
    else:
        df_ = df
        
    for i, colname in enumerate(df.columns):
        if len(axes.shape) > 1:
            ax = axes[row, col]
        else:
            ax = axes[i]
        ax.plot(df_[colname], color="k", linewidth=0.5)
        y1, y2 = ax.get_ylim()
        ax.fill_between(x = np.arange(0,burn_in), y1=y1, y2=y2, alpha=0.2)
        ax.set_ylim(y1, y2)
        ax.set_xlim(min(df_.index), max(df_.index))
        ax.set_ylabel(fancy_name(colname))
        
        col += 1
        if col % ncol == 0:
            row += 1
            col = 0
    fig.tight_layout()
    return fig


"""
Plot prior and posterior distributions
"""
def plot_prior_posterior_iid_exp(df, pl=0.2, pm=0.2, pq=(1, 1), pe=(1.5, 1), ncol=5, figsize=None):
    from math import ceil
    nrow = ceil(len(df.columns)/ncol)
    if not figsize:
        figsize=(12, 4*nrow)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    row, col = 0, 0

    for i, colname in enumerate(df.columns):
        if len(axes.shape) > 1:
            ax = axes[row, col]
        else:
            ax = axes[i]
        if colname == "lhood":
            prior = []
        elif colname == "prior":
            prior = []
        elif colname.startswith("l"):
            prior = np.random.exponential(pl, len(df[colname]))
        elif colname.startswith("m"):
            prior = np.random.exponential(pm, len(df[colname]))
        elif colname.startswith("q"):
            prior = np.random.beta(pq[0], pq[1], len(df[colname]))
        elif colname.startswith("eta"):
            prior = np.random.beta(pe[0], pe[1], len(df[colname]))
        else:
            continue
        
        if len(prior) > 0:
            bins, edges = np.histogram(np.hstack([prior, df[colname]]), bins=50)
            ax.hist(prior, color="k", linewidth=0.5, alpha=0.2, bins=edges)
            ax.hist(df[colname], bins=edges, color="cornflowerblue", alpha=0.5)
        else:
            ax.hist(df[colname], bins=50, color="cornflowerblue", alpha=0.5)
            
        ax.set_xlabel(fancy_name(colname))
        col += 1
        if col % ncol == 0:
            row += 1
            col = 0
    fig.tight_layout()
    sns.despine(offset=1)
    return fig


"""
Get fancy labels
"""
def fancy_name(s):
    if s == "lhood":
        s_ = "$L$"
    elif s[0] == "l":
        s_ = "$\lambda_{{{}}}$".format(s[1:])
    elif s[0] == "m":
        s_ = "$\mu_{{{}}}$".format(s[1:])
    elif s == "eta":
        s_ = "$\eta$"
    elif s == "\nu":
        s_ = "\nu"
    elif s[0] == "q":
        s_ = "$q_{{{}}}$".format(s[1:]) 
    else:
        s_ =  s
    return s_


"""
Autocorrelation function
"""
def acorr(x):
    x = x - x.mean()
    autocorr = np.correlate(x, x, mode='full')
    autocorr = autocorr[x.size:]
    autocorr /= autocorr.max()
    return autocorr


"""
Plot autocorrelation
"""
def plot_ac(df, burn_in=1000, ncol=2, figsize=None, discard=False):
    from math import ceil
    nrow = ceil(len(df.columns)/ncol)
    if not figsize:
        figsize=(12, 4*nrow)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    row, col = 0, 0
    
    if discard:
        df_ = df.loc[burn_in:]
    else:
        df_ = df
        
    for i, colname in enumerate(df.columns):
        if len(axes.shape) > 1:
            ax = axes[row, col]
        else:
            ax = axes[i]
        x = df[colname]
        ax.plot(df.iloc[1:].index, acorr(x), color="k")
        ax.fill_between(df.iloc[1:].index, 0, acorr(x), color="k", alpha=0.2)
        x1, x2 = ax.get_xlim()
        ax.hlines(y = 0, xmax=x2, xmin=x1, linestyles='--')        
        y1, y2 = ax.get_ylim()
        ax.fill_between(x = np.arange(0,burn_in), y1=y1, y2=y2, alpha=0.2)
        ax.set_ylim(y1, y2)
        #ax.set_xlim(min(df_.index), max(df_.index))
        ax.set_xlim(max(0, x1), x2)
        ax.set_ylabel(fancy_name(colname))
        
        col += 1
        if col % ncol == 0:
            row += 1
            col = 0
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1], index_col=0)
    fig = plot_trace(df, ncol=5, figsize=(14,12));
    fig.savefig(sys.argv[2] + ".trace.pdf")
    fig = plot_prior_posterior_iid_exp(df, pe=(20, 2), figsize=(14,12));
    fig.savefig(sys.argv[2] + ".prior-post.pdf")
    fig = plot_ac(df, ncol=5, figsize=(14,12));
    fig.savefig(sys.argv[2] + ".ac.pdf")
    
