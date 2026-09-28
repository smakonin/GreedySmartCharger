#
# libUtil.py -- Copyright (C) 2018 Stephen Makonin
#

import os


def limiter(need, production, limit):
    available = production
    if production > limit:
        available = limit
    if need < available:
        return need
    return available

def argmin(a):
    val = min(a)
    idx = a.index(val)
    return idx

def rmdup(k):
    new_k = []
    for elem in k:
        if elem not in new_k:
            new_k.append(elem)
    return new_k

def lbound(x, limit):
    if x < limit:
        return limit
    return x

def errout(text):
    print(text)
    print()
    exit(1)

def getfiles(path, prefix='', suffix=''):
    found = []
    for filename in os.listdir(path):
        if filename.startswith(prefix) and filename.endswith(suffix):
            found.append(filename)
    return found
