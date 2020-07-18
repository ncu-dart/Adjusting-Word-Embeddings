import argparse
import torch
from torch.utils.data import DataLoader
import os
import bcolz
import pickle
import operator
import scipy.stats
import numpy as np
from tqdm import tqdm
from model import BERT
from trainer import BERTTrainer
from dataset import BERTDataset, WordVocab

def LoadInputVector(wordvec, data, lookup):
    input1, input2 = [], []
    OOV = []
    for i in range(len(data[:,0])):
        try:
            input1.append(wordvec[data[i,0]])
            try:
                input2.append(wordvec[data[i,1]])
            except:
                input1.pop()
                print(wordvec[data[i,0]],wordvec[data[i,1]])
        except:
            print(wordvec[data[i,0]],wordvec[data[i,1]])

    return np.array(input1), np.array(input2)

def Evaluating_MEN(wordvec, data, lookup):
    input1, input2 = LoadInputVector(wordvec, data, lookup)
    output = []
    epsilon = 1e-5
    for i in range(len(input1)):
        output.append(np.dot(input1[i], input2[i])/(np.linalg.norm(input1[i])*np.linalg.norm(input2[i])))
    output = (np.array(output)).reshape(-1)
    return round(scipy.stats.spearmanr(output, np.array(data[:,2], dtype=float))[0], 4)

def print_word_vecs(wordVectors, outFileName):
    print("Writing down the vectors in", outFileName)
    outFile = open(outFileName, 'w', encoding='utf-8')  
    for word, values in wordVectors.items():
        outFile.write(word+' ')
        for val in wordVectors[word]:
            outFile.write('%.4f' %(val)+' ')
        outFile.write('\n')      
    outFile.close()

parser = argparse.ArgumentParser()

parser.add_argument("-ep", "--emb_path", required=True, type=str, help="filepath of the original pre-trained embeddings")
parser.add_argument("-vp", "--vocab_path", required=True, type=str, help="filepath of vocabulary")
parser.add_argument("-op", "--output_path", required=True, type=str, help="filepath for saving embeddings")

args = parser.parse_args()

# Read Vocab
print("Loading Vocab", args.vocab_path)
vocab = WordVocab.load_vocab(args.vocab_path)
print("Vocab Size: ", len(vocab))

word2idx = vocab.stoi
idx2word = vocab.itos


# Read WordVec
NormRead = True
wordVecs = {}
with open(args.emb_path, 'r', encoding='utf-8') as fileObject:
    for line in fileObject:
        tokens = line.strip().lower().split()
        try:
            wordVecs[tokens[0]] = np.fromiter(map(float, tokens[1:]), dtype=np.float64)
            if NormRead:
                wordVecs[tokens[0]] = wordVecs[tokens[0]] / np.sqrt((wordVecs[tokens[0]]**2).sum() + 1e-5)
        except:
            pass


# Read Model Raw Output
new_wordVecs = {}
for key,val in wordVecs.items():
    new_wordVecs[key] = val

import pickle
for i in tqdm(range(int(len([name for name in os.listdir('../output/embeddings/raw/')])/2))):
    inp = pickle.load(open('../output/embeddings/raw/result_input_iter{}.pkl'.format(i),'rb')).cpu().numpy()
    out = pickle.load(open('../output/embeddings/raw/result_output_iter{}.pkl'.format(i),'rb')).cpu().detach().numpy()
    for j in range(inp.shape[0]):
        for x,y in zip(inp[j],out[j]):
            try:
                new_wordVecs[idx2word[x]] = np.vstack((new_wordVecs[idx2word[x]],y))
            except:
                pass

for key,val in new_wordVecs.items():
    if len(val.shape)>1:
        new_wordVecs[key] = (sum(val)-wordVecs[key])/val.shape[0]

if NormRead:
    for key,val in new_wordVecs.items():
        new_wordVecs[key] = val / np.sqrt((val**2).sum() + 1e-5)


#Word Sim Task
tasks = ['MEN_3k', 'SL_999', 'WS_353', 'RG_65']
for i in tasks:
    with open('../data/testsets/{}.txt'.format(i), 'r', encoding='utf-8') as fp_men:
        fp_men_ = fp_men.readlines()
        data_men = [row.strip().split(' ') for row in fp_men_]
        data_men = np.array(data_men)

    word_to_idx_men = {}
    idx = 0

    for w in data_men[:,0]:
        try: word_to_idx_men[w]
        except KeyError:
            word_to_idx_men[w] = idx
            idx = idx+1
            
    for w in data_men[:,1]:
        try: word_to_idx_men[w]
        except KeyError:
            word_to_idx_men[w] = idx
            idx = idx+1
    
    word_to_idx_men = sorted(word_to_idx_men.items(), key=operator.itemgetter(1))
    lookup_men = dict(word_to_idx_men)

    print('<{} Dataset>'.format(i))
    print("Before :", Evaluating_MEN(wordVecs, data_men, lookup_men))
    print("After  :", Evaluating_MEN(new_wordVecs, data_men, lookup_men))

print_word_vecs(new_wordVecs, args.output_path)
