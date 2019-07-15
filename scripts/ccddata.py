#!/usr/bin/env python3
# coding: utf-8
# author: Arthur Zwaenepoel

import os
import sys
if len(sys.argv) != 2:
    print("Usage: {0} <ale_dir>".format(sys.argv[0]))
    exit()

aledir = sys.argv[1]
ale = [os.path.join(aledir, fname) for fname in os.listdir(aledir)]

def aledata(fname):
    with open(fname, "r") as f:
        content = f.read().split("#")
        nclades = len(content[3].split("\n")[1:-1])
        ntriples = len(content[5].split("\n")[1:-1])
        ntax = len(content[7].split("\n")[1:-1])
    return (ntax, nclades, ntriples)

print("orthogroup,ntaxa,nclades,ntriples")
for f in ale:
    print(os.path.abspath(f), end=",")
    print(",".join([str(x) for x in aledata(f)]))
