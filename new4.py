# Seems to work, but could be optimized using tensordot/bmm instead of naive multiplications

# This one (new4.py) samples multiple corruptions, and performs the full inference for each

# Currently the problem is that the pruning method has low precision (lets through many bad prefixes), and that the current heuristic leads yto very long hypothesis that slow down inference. A regular expression would be a good solution.

import argparse
parser = argparse.ArgumentParser()


parser.add_argument("--BATCHSIZE", type=int, default=100)
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

NOISE_RATE = 0.1

p_mod = 0.1
p_rc = 0.3
p_src = 0.8 # 1e-5 # 0.0 for German, 0.8 for English
################################

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

validBigrams = set()
validBigrams.add(("N", "V"))
validBigrams.add(("N", "C"))
validBigrams.add(("N", "P"))

validBigrams.add(("V", "N"))
validBigrams.add(("V", "V"))

validBigrams.add(("C", "V"))
validBigrams.add(("C", "N"))

validBigrams.add(("P", "N"))

badTrigrams = set()
badTrigrams.add(("N", "V", "N"))
#badTrigrams.add(("V", "N", "V"))

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
print(matrixLeft)
print(matrixLeft.sum(dim=1))
invertedLeft = torch.inverse(matrixLeft)
print(invertedLeft)














sentence = ["N", "C", "N", "C", "N", "V", "V"] #, "V"] # , "."
#sentence = ["N", "C", "N", "V", "V"] # , "."


def corrupt(s):
   return [x for x in s if random.random() > NOISE_RATE]

print(corrupt(sentence))
print(corrupt(sentence))
print(corrupt(sentence))
print(corrupt(sentence))
print(corrupt(sentence))
print(corrupt(sentence))
print(corrupt(sentence))
print(corrupt(sentence))


# a proposal distribution that 
def inverseCorrupt(s):
   while True:
      sa = []
      loglikelihood = 0
      for i in range(len(s)+1):   
         while random.random() < 0.3 or (len(sa) > 0 and i < len(s) and (sa[-1], s[i]) not in validBigrams):
            loglikelihood += log(0.1)
            nextSymbol = random.choice(terminals)
            while (len(sa) > 0 and (sa[-1], nextSymbol) not in validBigrams) or (len(sa) == 0 and nextSymbol in [".", "C", "P", "V"]) or (len(sa) > 1 and (tuple(sa[-2:])+(nextSymbol,)) in badTrigrams):
                nextSymbol = random.choice(terminals)
            sa.append(nextSymbol)
         loglikelihood += log(0.9)
         if i < len(s):
            sa.append(s[i])
      fail=False
      if sa[0] in [".", "C", "P", "V"]:
        fail=True
      vs = len([x for x in sa if x == "V"])
      ns = len([x for x in sa if x == "N"])
      if vs > ns:
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
for corruptionInd in range(100):
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
         proposals.append((list(sa), loglik))
         proposalsSet.add(sa)
#         print(len(proposals))
     
    print(["".join(x) for x in proposalsSet])
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
         lengths = [len(x) for x in sentences]
         maxLen = max(lengths)
         sentences = [["PAD" for _ in range(maxLen-len(x))]+x for x in sentences]
         
         inputs = torch.LongTensor([[stoi[x] if x in stoi else -1 for x in sentence] for sentence in sentences]).view(args.BATCHSIZE, -1).t() # sequenceLength x BATCHSIZE
         #print(inputs)
         
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
                   for i in range(args.BATCHSIZE):
                      ((chart[start, length, 0, i , inputs[start,i]])) = 0
    #               print(inputs[start,:])
    #               print(((chart[start, length, 0, : , :])))
    #               print("Penultimate")
    #               print(((chart[start, length, 0, -2 , :])))
    #               print("Ultimate")
    #               print(((chart[start, length, 0, -1 , :])))
    #
    #               if start == sequenceLength-1:
    #                  quit()
                
         #          print(((chart[start, length, HEIGHTS, : , inputs[start,:]])))
         #          print(((chart[start, length, HEIGHTS])))
         #
         #          print(unary_productions.size(), ((chart[start, length, HEIGHTS, : , :])).size())
         #
         #
         #          print(torch.exp(((chart[start, length, HEIGHTS, : , :]))))
         #          print((torch.exp(unary_productions.unsqueeze(0).expand(args.BATCHSIZE, -1, -1)) * torch.exp(((chart[start, length, HEIGHTS, : , :])).unsqueeze(1).expand(-1, TERMINALS+NONTERMINALS, -1))).sum(dim=2))
         
                   ((chart[start, length, 0, : , :])) = torch.log(torch.exp(((chart[start, length, 0, : , :]))) + (torch.exp(unary_productions.unsqueeze(0).expand(args.BATCHSIZE, -1, -1)) * torch.exp(((chart[start, length, 0, : , :])).unsqueeze(1).expand(-1, TERMINALS+NONTERMINALS, -1))).sum(dim=2))
    
                   assert ((chart[start, length, 0, : , :])).max() < 0.1, ((chart[start, length, 0, : , :]))
         #          print(((chart[start, length, HEIGHTS, : , :])))
                else:
                   results = [[] for _ in range(1)]
                   for intermediate in range(start+1, end+1): # start of the second constituent
                      assert intermediate <= end
                      height = 0
                      logprobsFromLeft = chart[start, intermediate-start-1, 0, :, :]
                      logprobsFromRight = chart[intermediate, end-intermediate, 0, :, :]
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
                      results[0].append(torch.log(fullProd) + rightMax + leftMax)
                      assert results[0][-1].max() < -1e-5, results[0][-1] # sometimes an assertion error is triggered here
                   if len(results[0]) == 0:
                      continue
                   resultsForHeight = torch.stack(results[0])
                   resMax = resultsForHeight.max()
                   if float(resMax) == float("-inf"):
                        continue
                   chart[start, end-start, 0, :, :NONTERMINALS] = torch.log(torch.exp(resultsForHeight - resMax).sum(dim=0)) + resMax
                   assert chart[start, end-start, 0, :, :NONTERMINALS].max() < -1e-5
            
         # first fill the prefix chart at the last element
      #   print(chart[sequenceLength-1, 0, 0, :, :])
         chartPrefix = torch.zeros(inputs.size()[0], 0+1, args.BATCHSIZE, (TERMINALS+NONTERMINALS))
         chartPrefix.fill_(float("-Inf"))
    
    
    
     #    print(chart[sequenceLength-1, 0, 0, :, :]) 
         assert chart[sequenceLength-1, 0, 0, :, :].max() < 0.1, chart[sequenceLength-1, 0, 0, :, :] 
       
    #     print(invertedLeft)
   #      print(invertedLeft.size(), chart[sequenceLength-1, 0, 0].size())
    
    
  #       print(invertedLeft.unsqueeze(0).expand(args.BATCHSIZE, -1, -1).size(), torch.exp(chart[sequenceLength-1, 0, 0, :, :]).unsqueeze(1).expand(-1, NONTERMINALS+TERMINALS, -1).size())
 #        print((invertedLeft.unsqueeze(0).expand(args.BATCHSIZE, -1, -1) * torch.exp(chart[sequenceLength-1, 0, 0, :, :]).unsqueeze(1).expand(-1, NONTERMINALS+TERMINALS, -1)).size())
#         print(chart[sequenceLength-1, 0, 0, :, :])
    #     quit()
         chartPrefix[sequenceLength-1, 0, :, :] = torch.log((invertedLeft.unsqueeze(0).expand(args.BATCHSIZE, -1, -1) * torch.exp(chart[sequenceLength-1, 0, 0, :, :]).unsqueeze(1).expand(-1, NONTERMINALS+TERMINALS, -1)).sum(dim=2))
   #      print(invertedLeft)
  #       print("OFFENDING CHART LINE")
 #        print(chart[sequenceLength-1, 0, 0, -1])
#         print(itos)
         assert chartPrefix[sequenceLength-1, 0, -1, :NONTERMINALS].max() < 0.1, chartPrefix[sequenceLength-1, 0, -1, :NONTERMINALS]
         assert chartPrefix[sequenceLength-1, 0, -2, :NONTERMINALS].max() < 0.1, chartPrefix[sequenceLength-1, 0, -2, :NONTERMINALS]
    
         assert chartPrefix[sequenceLength-1, 0, :, :NONTERMINALS].max() < 0.1, chartPrefix[sequenceLength-1, 0, :, :NONTERMINALS]
    
         assert chartPrefix[sequenceLength-1, 0, :, :].max() < 0.1, chartPrefix[sequenceLength-1, 0, :, :]
    #     print( chartPrefix[sequenceLength-1, 0, :, :])
     #    quit()
         for start in range(sequenceLength-2, -1, -1):
            results = [[] for _ in range(1)]
            for intermediateStart in range(start+1, sequenceLength):
               #print(start, intermediateStart, start+1, sequenceLength)
               logprobsFromLeft = chart[start, intermediateStart-start-1, 0, :, :]
               logprobsFromRight = chartPrefix[intermediateStart, 0, :, :]
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
               results[0].append(torch.log(fullProd) + rightMax + leftMax)
        #       print(torch.argmax(results[0][-1]), results[0][-1].size(), results[0][-1][-1][-1]) # TODO figure out what's going on here
               assert results[0][-1].max() < 1e-5, results[0][-1] # sometimes an assertion error is triggered here
            if len(results[0]) == 0:
               continue
            resultsForHeight = torch.stack(results[0])
            resMax = resultsForHeight.max()
            if float(resMax) == float("-inf"):
                 continue
            preliminaryResult = torch.log(torch.exp(resultsForHeight - resMax).sum(dim=0)) + resMax
         
       #     print(invertedLeft.size(), preliminaryResult.size())
            chartPrefix[start, 0, :, :NONTERMINALS] = torch.log((invertedLeft[:NONTERMINALS, :NONTERMINALS].unsqueeze(0).expand(args.BATCHSIZE, -1, -1) * torch.exp(preliminaryResult).unsqueeze(1).expand(-1, NONTERMINALS, -1)).sum(dim=2))
            assert chartPrefix[start,  0, :, :NONTERMINALS].max() < 1e-5
         
      #   print(sentences[:5])
     #    print(lengths)
    #     print("Prefix probability")
   #      print(chartPrefix[:, 0, :, stoi["ROOT"]])
         return torch.FloatTensor([chartPrefix[maxLen-lengths[i], 0, i, stoi["ROOT"]] for i in range(args.BATCHSIZE)])
       
    
    
    prefixProb = (parse([x[0] for x in proposals]))
    updatedProbEOS = (parse([x[0]+["."] for x in proposals]))
    updatedProbV = (parse([x[0]+["V"] for x in proposals]))
    
    print("WELLFORMED PREFIXES", ((prefixProb > float("-inf")).float().sum()))
    #print(updatedProb)
    #surprisal = (prefixProb - updatedProb)
    #print(torch.exp(-surprisal))
    #print(torch.exp(logImportanceWeights))
  #  print(torch.exp(logPCorruptedSource))
 #   print(torch.exp(logPCorruptedSource).sum())
    
    logNormalizationConstant = torch.log(torch.sum(torch.exp(prefixProb + logPCorruptedSource)))
#    print(logNormalizationConstant)
    
    marginalVerbProbability = torch.exp(logPCorruptedSource + updatedProbV -  logNormalizationConstant).sum()
    print("Marginal verb prob", marginalVerbProbability)
    
    
    marginalEOSProbability = torch.exp(logPCorruptedSource + updatedProbEOS -  logNormalizationConstant).sum()
    print("Marginal EOS prob", marginalEOSProbability)
    surprisalsV.append(log(1e-10+float(marginalVerbProbability)))
    surprisalsEOS.append(log(1e-10+float(marginalEOSProbability)))
    print("SURP VERB", surprisalsV)
    print("SURP EOS", surprisalsEOS)
    print("SURP Avg VERB", -sum([surprisalsV[i] for i in range(len(surprisalsV))])/len(surprisalsV))
    print("SURP Avg EOS", -sum([surprisalsEOS[i] for i in range(len(surprisalsV))])/len(surprisalsV))
