# Based on new4.py, but does simple toy center embedding

import argparse
parser = argparse.ArgumentParser()


parser.add_argument("--BATCHSIZE", type=int, default=1000)
args=parser.parse_args()
print(args)

import random
random.seed(0)

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

################################
# NOTE: The parameters are different from those used in the original paper.

NOISE_RATE = 0.2

p_mod = 0.1
p_rc = 0.3
p_src = 0.8 # 1e-5 # 0.0 for German, 0.8 for English

p_rec = 0.3
################################

# unary rules are OK for terminals
rules = [
    (("ROOT", ("S", ".")), 1),
    (('S', ('A1', 'S1')), 0.33 * p_rec),
    (('S', ('A2', 'S2')), 0.33 * p_rec),
    (('S', ('A3', 'S3')), 0.33 * p_rec),
#    (('S', ('A4', 'S4')), 0.2 * p_rec),
 #   (('S', ('A5', 'S5')), 0.2 * p_rec),
    (('S', ('A1', 'B1')), 0.33 * (1-p_rec)),
    (('S', ('A2', 'B2')), 0.33 * (1-p_rec)),
    (('S', ('A3', 'B3')), 0.33 * (1-p_rec)),
#    (('S', ('A4', 'B4')), 0.2 * (1-p_rec)),
 #   (('S', ('A5', 'B5')), 0.2 * (1-p_rec)),
    (('S1', ('S', 'B1')), 1),
    (('S2', ('S', 'B2')), 1),
    (('S3', ('S', 'B3')), 1),
  #  (('S4', ('S', 'B4')), 1),
   # (('S5', ('S', 'B5')), 1),
]

validBigrams = set()
for i in range(1,6):
  for j in range(1, 6):
      validBigrams.add(("A"+str(i), "A"+str(j)))
      validBigrams.add(("B"+str(i), "B"+str(j)))
      validBigrams.add(("A"+str(i), "B"+str(j)))

badTrigrams = set()

terminals = set(["A"+str(i) for i in range(1,6)] + ["B"+str(i) for i in range(1,6)])
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

terminals = [x for x in itos if x not in nonterminals]

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


matrixLeft = torch.FloatTensor([[0 for _ in range(NONTERMINALS+TERMINALS)] for _ in range(NONTERMINALS+TERMINALS)]) # traces the LEFT edge
for rule, prob in rules:
  left, right = rule
  if len(right) == 1:
    continue
  matrixLeft[stoi[left]][stoi[right[0]]] -= prob


for i in range(NONTERMINALS+TERMINALS):
    matrixLeft[i][i] += 1
#print(matrixLeft)
#print(matrixLeft.sum(dim=1))
invertedLeft = torch.inverse(matrixLeft)
#print(invertedLeft)














sentence = ["A1", "A2", "A3", "B3"] #, "V"] # , "."
#sentence = ["N", "C", "N", "V", "V"] # , "."


def corrupt(s):
   return [x for x in s if random.random() > NOISE_RATE]

#print(corrupt(sentence))
#print(corrupt(sentence))
#print(corrupt(sentence))
#print(corrupt(sentence))
#print(corrupt(sentence))
#print(corrupt(sentence))
#print(corrupt(sentence))
#print(corrupt(sentence))
#
def updateStack(stack, symbol):
   if symbol.startswith("A"):
      stack.append(symbol)
   else:
      del stack[-1]

# a proposal distribution that 
def inverseCorrupt(s):
   while True:
      sa = []
      stack = []
      loglikelihood = 0
      fail=False

      for i in range(len(s)+1):   
         while random.random() < 0.3 or (len(sa) > 0 and i < len(s) and (sa[-1], s[i]) not in validBigrams):
            loglikelihood += log(0.1)
            nextSymbol = random.choice(terminals) #  or (len(sa) == 0 and nextSymbol in [".", "C", "P", "V"])
            if len(sa) > 0 and len(stack) == 0:
               return tuple(sa), 0         
            while nextSymbol == "." or (len(sa) > 0 and (sa[-1], nextSymbol) not in validBigrams) or (len(sa) > 1 and (tuple(sa[-2:])+(nextSymbol,)) in badTrigrams) or (i < len(s) and (nextSymbol.startswith("B") or nextSymbol == ".") and s[i].startswith("A")) or (nextSymbol.startswith("B") and (len(stack) == 0 or nextSymbol[1] != stack[-1][1])):
 #               if len(sa) > 0:
#                   print("Trying next symbol", nextSymbol, (sa[-1], nextSymbol, sa[-5:]))
                nextSymbol = random.choice(terminals)
            sa.append(nextSymbol)
            updateStack(stack, nextSymbol)
         loglikelihood += log(0.9)
         if i < len(s):
            sa.append(s[i])
            if len(stack) == 0 and s[i].startswith("B"):
               fail=True
               break
            updateStack(stack, s[i])

    #  print("Trying", sa)
#      if sa[0] in [".", "C", "P", "V"]:
 #       fail=True
      as_ = len([x for x in sa if x.startswith("A")])
      bs_ = len([x for x in sa if x.startswith("B")])
      if len(sa) == 0:
        fail = True
      if bs_ > as_:
#        print(sa)
        fail=True
      for j in range(len(sa)-1):
         if fail:
            break
         if tuple(sa[j:j+2]) not in validBigrams:
             fail = True
      if not fail:
        return tuple(sa), loglikelihood
 #     else:
#         print("Reject", sa)

surprisalsV = []
surprisalsEOS = []
for corruptionInd in range(1000):
    corrupted = []
    while len(corrupted) == 0:
       corrupted = corrupt(sentence)
    print("CORRUPTED")
    print(corrupted)
#    for _ in range(10):
#      print(inverseCorrupt(corrupted))
    
    proposalsSet = set()
    proposals = []
    while len(proposals) < args.BATCHSIZE:
      sa, loglik = inverseCorrupt(corrupted)
      if sa not in proposalsSet:
         #print(sa)
         proposals.append((list(sa), loglik))
         proposalsSet.add(sa)
#         print(len(proposals))
     
    print(["-".join(x) for x in proposalsSet])
    s = [x[0] for x in proposals]
    probs = torch.FloatTensor([x[1] for x in proposals])
    lengths = torch.LongTensor([len(x) for x in s])
    maxLen = max(lengths)
    s = [x+["PAD" for _ in range(maxLen-len(x))] for x in s]
    s = torch.LongTensor([[stoi[x] if x in stoi else -1 for x in y] for y in s])
#    print(s)
    t = [stoi[x] for x in corrupted] # target
#    print("Source", s)
#    print("Target", t)



    # LEVENSHTEIN ALGORITHM in sum-semiring
    v0 = torch.FloatTensor(len(proposals), maxLen+1)
    v1 = torch.FloatTensor(len(proposals), maxLen+1)
    
    # start by assuming empty target
    for i in range(maxLen+1):
        v0[:,i] = i * log(NOISE_RATE)
    
    for i in range(0, len(t)):
        # consider empty source
        v1[:,0] = float("-inf") # source[:0] vs target[:i+1]
    
        for j in range(0, maxLen): # consider source[:j+1] vs target[:i+1]
                                   # maybe source[j] and target[i] come from the same source, so we can go back to source[:j] and target[:i]
                                   # maybe source[j] was deleted, so we can go back to source[:j] and target[:i+1]
            insertionCost = v1[:, j] + log(NOISE_RATE) 
            substitutionCost = torch.FloatTensor(len(proposals))
            substitutionCost.fill_(float("-inf"))
            relevantBatches = (s[:,j] == t[i])
            substitutionCost[relevantBatches] = v0[relevantBatches,j] + log(1-NOISE_RATE)
    
            v1[:, j + 1] = torch.log(torch.exp(insertionCost) + torch.exp(substitutionCost))
#        print(i, "v1", v1)
        v0, v1 = v1, v0
#    print(v0.size(), lengths.size())
#    print(lengths)
    result = torch.FloatTensor([v0[i,lengths[i]] for i in range(len(proposals))]) # log P(corrupted|source)
#    print("logConditionals")
#    print(result)
    
    logPCorruptedSource = result
#    print("sum of all conditionals", torch.exp(logPCorruptedSource).sum())
#    print(len(proposalsSet))
#    print(proposals[0])
#    print(proposals[1])
#    print(proposals[2])
#    print(proposals[3])
#    print(proposals[4])
    
#    print(corrupted)
    
    
    # log-importance weights
    #logImportanceWeights = result - probs
#    print(logImportanceWeights)
    
    # CHART PARSER for prefix probabilities (could be made a global function)
    def parse(sentences):
       likelihoods = [0 for _ in sentences]
       stacks = [[] for _ in sentences]
       for i, sent in enumerate(sentences):
          stack = stacks[i]
          for j, word in enumerate(sent):
             assert word != "."
             if word.startswith("A"):
               stack.append(word)
             elif len(stack) == 0:
               likelihoods[i] = float("-inf")
  #             print(sent, "Empty stack", word)
               break
             elif stack[-1][1] == word[1]:
               del stack[-1]
             else:
 #              print(sent, "Mismatch")
               likelihoods[i] = float("-inf")
               break
             if word.startswith("A"):
                likelihoods[i] += log(0.33 * p_rec)
             elif sent[j-1].startswith("A"):
                likelihoods[i] += log(1- p_rec)
#       print(likelihoods)
 #      quit()
       return torch.FloatTensor(likelihoods)   
    
    
    prefixProb = (parse([x[0] for x in proposals]))
    updatedProbEOS = (parse([x[0]+["B2"] for x in proposals]))
    updatedProbV = (parse([x[0]+["B1"] for x in proposals]))
    print(prefixProb)    
    print("WELLFORMED PREFIXES", ((prefixProb > float("-inf")).float().sum()))
#    assert    ((prefixProb > float("-inf")).float().sum()) >= args.BATCHSIZE

    logNormalizationConstant = torch.log(torch.sum(torch.exp(prefixProb + logPCorruptedSource)))
    
    marginalVerbProbability = torch.exp(logPCorruptedSource + updatedProbV -  logNormalizationConstant).sum()
    print("Marginal verb prob", marginalVerbProbability)
    
    marginalEOSProbability = torch.exp(logPCorruptedSource + updatedProbEOS -  logNormalizationConstant).sum()
    print("Marginal EOS prob", marginalEOSProbability)
    surprisalsV.append(log(1e-10+float(marginalVerbProbability)))
    surprisalsEOS.append(log(1e-10+float(marginalEOSProbability)))
 #   print("SURP VERB", surprisalsV)
#    print("SURP EOS", surprisalsEOS)
    print("SURP Avg VERB", -sum([surprisalsV[i] for i in range(len(surprisalsV))])/len(surprisalsV))
    print("SURP Avg EOS", -sum([surprisalsEOS[i] for i in range(len(surprisalsV))])/len(surprisalsV))
#    quit()

