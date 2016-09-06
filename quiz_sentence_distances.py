#!/usr/bin/python

import numpy as np
from collections import Counter
import math

def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator

sa = "The quick brown fox jumps over the lazy dog".lower()
sb = "A quick brown dog outpaces a quick fox".lower()

a = sa.split(" ")
b = sb.split(" ")

wdict = np.unique(a+b)
print wdict

ca = Counter(a)
cb = Counter(b)

d = 0
print "Key\tva\tvb\(va-vb)**2"
for k in wdict:
    print k,"\t",

    if k in ca:
        va = ca[k]
    else:
        va = 0

    print va,"\t",

    if k in cb:
        vb = cb[k]
    else:
        vb = 0

    print vb,"\t",

    d += (va-vb)**2

    print (va-vb)**2,"\t"

print "Euclidian distance is: %.3f" % math.sqrt(d)
print "Cosine distance is: %.3f" % (1-get_cosine(ca,cb))
