import argparse
parser = argparse.ArgumentParser()


parser.add_argument("--BATCHSIZE", type=int, default=1)
args=parser.parse_args()
print(args)



from math import log

#import functools
#import itertools
#
#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
#import networkx as nx
#import sympy
#
#
#from pmonad import *
#import incnoise
#import infotrees
#import pcfg
#
EPSILON = 10 ** -4

def is_close(x, y, tol):
    return abs(x - y) < tol

p_mod = 0.2
p_rc = 0.3
p_src = 0.8

# unary rules are OK for terminals
rules = [
    (("ROOT", ("S", ".")), 1),
    (('S', ('NP', 'V')), 1),
    (('NP', ('N',)), 1 - p_mod),
    (('NP', ('N', 'RC')), p_mod * p_rc),
    (('NP', ('N', 'PP')), p_mod * (1 - p_rc)),
    (('PP', ('P', 'NP')), 1),
    (('RC', ('C', 'S')), 1 - p_src),
    (('RC', ('C', 'VNP')), p_src),
    (('VNP', ('V', 'NP')), 1)
]

terminals = set(["N", "C", "V", "P"])
symbols = set()
nonterminals = set()
for rule, prob in rules:
   left, right = rule
   symbols.add(left)
   assert len(right) <= 2
   for cat in right:
      symbols.add(cat)
   nonterminals.add(left)


itos = sorted(list(symbols), key=lambda x:(x not in nonterminals, x))
stoi = dict(list(zip(itos, range(len(itos)))))
NONTERMINALS = len(nonterminals)
TERMINALS = len(itos) - NONTERMINALS
print(itos)

import torch

binary_productions = torch.FloatTensor(NONTERMINALS, NONTERMINALS+TERMINALS, NONTERMINALS+TERMINALS)
binary_productions.fill_(float("-inf"))

unary_productions = torch.FloatTensor(NONTERMINALS+TERMINALS, NONTERMINALS+TERMINALS)
unary_productions.fill_(float("-inf"))

for rule, prob in rules:
   left, right = rule
   assert len(right) <= 2
   if len(right) == 2:
      binary_productions[stoi[left]][stoi[right[0]]][stoi[right[1]]] = log(prob)
   elif len(right) ==1:
      unary_productions[stoi[left]][stoi[right[0]]] = log(prob)
   else:
      assert False
#for j in range(TERMINALS):
#    unary_productions[j+NONTERMINALS][j+NONTERMINALS] = 0






sentence = ["N", "C", "N", "C", "N", "V", "V", "V", "."]

inputs = torch.LongTensor([stoi[x] for x in sentence]).view(args.BATCHSIZE, -1).t() # sequenceLength x BATCHSIZE
print(inputs)

HEIGHTS=0
sequenceLength = inputs.size()[0]
chart = torch.zeros(inputs.size()[0], inputs.size()[0], 0+1, args.BATCHSIZE, (TERMINALS+NONTERMINALS))
chart.fill_(float("-Inf"))
binary_productionsHere = binary_productions
for length in range(sequenceLength):
   for start in range(sequenceLength): # first word
       end = start+length # last word
       if end >= sequenceLength:
          continue
       if length == 0: # a single word
#          print("==============")
#          print(inputs[start,:])
          ((chart[start, length, 0, : , inputs[start,:]])) = 0
#          print(((chart[start, length, HEIGHTS, : , inputs[start,:]])))
#          print(((chart[start, length, HEIGHTS])))
#
#          print(unary_productions.size(), ((chart[start, length, HEIGHTS, : , :])).size())
#
#
#          print(torch.exp(((chart[start, length, HEIGHTS, : , :]))))
#          print((torch.exp(unary_productions.unsqueeze(0).expand(args.BATCHSIZE, -1, -1)) * torch.exp(((chart[start, length, HEIGHTS, : , :])).unsqueeze(1).expand(-1, TERMINALS+NONTERMINALS, -1))).sum(dim=2))

          ((chart[start, length, 0, : , :])) = torch.log(torch.exp(((chart[start, length, 0, : , :]))) + (torch.exp(unary_productions.unsqueeze(0).expand(args.BATCHSIZE, -1, -1)) * torch.exp(((chart[start, length, 0, : , :])).unsqueeze(1).expand(-1, TERMINALS+NONTERMINALS, -1))).sum(dim=2))
#          print(((chart[start, length, HEIGHTS, : , :])))
       else:
          results = [[] for _ in range(1)]
          for intermediate in range(start+1, end+1): # start of the second constituent
             assert intermediate <= end
             height = 0
             logprobsFromLeft = chart[start, intermediate-start-1, height, :, :]
             logprobsFromRight = chart[intermediate, end-intermediate, height, :, :]
             rightMax = logprobsFromRight.max()
             if float(rightMax) == float("-inf"):
                continue
             prodWithRight = torch.exp(binary_productionsHere.unsqueeze(0).expand(args.BATCHSIZE, -1, -1, -1)) * torch.exp(logprobsFromRight.unsqueeze(1).unsqueeze(1).expand(-1, NONTERMINALS, TERMINALS+NONTERMINALS, -1) - rightMax)
             prodWithRight = prodWithRight.sum(dim=3)
             leftMax = logprobsFromLeft.max()
             if float(leftMax) == float("-inf"):
                continue
             fullProd = prodWithRight * torch.exp(logprobsFromLeft.unsqueeze(1).expand(-1, NONTERMINALS, -1) - leftMax)
             fullProd = fullProd.sum(dim=2)
             results[height].append(torch.log(fullProd) + rightMax + leftMax)
             assert results[height][-1].max() < -1e-5, results[height][-1] # sometimes an assertion error is triggered here
          height = 0
          if len(results[height]) == 0:
             continue
          resultsForHeight = torch.stack(results[height])
          resMax = resultsForHeight.max()
          if float(resMax) == float("-inf"):
               continue
          chart[start, end-start, height, :, :NONTERMINALS] = torch.log(torch.exp(resultsForHeight - resMax).sum(dim=0)) + resMax
          assert chart[start, end-start, height, :, :NONTERMINALS].max() < -1e-5
overallProbabilities = torch.zeros(args.BATCHSIZE)
covered = torch.zeros(args.BATCHSIZE).byte()
CHART_END = chart[0, :, 0, :, stoi["ROOT"]]
print(CHART_END)
for i in range(CHART_END.size()[0]-1, -1, -1):
   coveredHere = (CHART_END[i] > float("-inf"))
   toBeAdded = (coveredHere * (1-covered))
   covered = covered + toBeAdded
   overallProbabilities = torch.where(toBeAdded, CHART_END[i], overallProbabilities)

