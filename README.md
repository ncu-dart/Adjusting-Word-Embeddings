# Adjusting Word Embeddings

# Introduction
A post-training and listwise method for adjusting word embeddings with synonyms and antonyms based on undirected list encoder generated from self-ettention.

Paper Links: [https://drive.google.com/file/d/1UO6FVOwuAenNfWnkFrylgzHQe-9WQkff/view](https://drive.google.com/file/d/1UO6FVOwuAenNfWnkFrylgzHQe-9WQkff/view)

# Nitty-gritty to Know Before Starting

## Encoding of Data

- Please make sure the .txt files of embeddings and lexicons that you're introducing to the model are encoded in UTF-8. 
- Performance may drop significantly if .txt files are not uniformly encoded. This is an empirical conclusion.
- Simply way to make a UTF-8 encoding txt file (on a Windows computer):
  - Open the target txt file (that you're not sure if UTF-8 encoded) in Windows Wordpad.
  - Press File -> Save As -> choose "UTF-8" at the Encoding box -> Save

## Loss Function

- The loss funtions locate in ```./code/trainer/pretrain.py```.
- The proposed loss function in the paper will not lead to the experiment results in the paper. The corresponding python code of this code 
  is commented in the .py file.
  - The proposed loss function is 
- To reproduce the experiment results stated in the paper, please ......



# Quickstart

## main<span></span>.py
* Train self-attention model and adjust pre-trained word embeddings.

Usage:
```
$python main.py -ep <filepath of pre-trained embeddings> 
                -en <filename of pre-trained embeddings> 
                -lp <filepath of lexicons>
                -vp <filepath of vocabulary>
                -op <filepath to save model>
```
Example:
```
$python main.py -ep ../data/embeddings/GloVe/glove.6B.300d.txt 
                -en glove.6B.300d 
                -lp ../data/lexicons/wordnet_syn_ant.txt 
                -vp ../data/embeddings/GloVe/glove.6B.300d.txt.vocab.pkl 
                -op ../output/model/listwise.model
```

## evaluation<span></span>.py
* Will evaluate the lastly trained model and generate a .txt file for the adjusted embedding at ```<filepath for saving embeddings>```.
* Compare the performance of word embeddings on word similarity tasks before and after adjusting.  
* Process the raw output to GloVe format.

Usage:
```
$python evaluation.py -ep <filepath of the original pre-trained embeddings>
                      -vp <filepath of vocabulary>
                      -op <filepath for saving embeddings>
```
Example:
```
$python evaluation.py -ep ../data/embeddings/GloVe/glove.6B.300d.txt 
                      -vp ../data/embeddings/GloVe/glove.6B.300d.txt.vocab.pkl
                      -op ../output/embeddings/Listwise_Vectors.txt
```

# Datasets

## Embeddings
* Pretrained word embeddings filtered by 50K frequent words in GloVe format.

Data format:
```
word1 -0.09611 -0.25788 ... -0.092774  0.39058
word2 -0.24837 -0.45461 ...  0.15458  -0.38053
```

## Lexicons
* Synonyms and antonyms retrieved from dictionary.

Data format:
```
word1 syn1 ... synn \t ant1 ... antn 
word2 syn1 ... synn \t ant1 ... antn 
```

## Testsets
* Similarity tasks datasets.

Data format:
```
word1 word2 50.00
word1 word2 49.00
```

# Reference
1. https://github.com/mfaruqui/retrofitting
2. https://github.com/nmrksic/counter-fitting
3. https://github.com/HwiyeolJo/Extrofitting
4. https://github.com/codertimo/BERT-pytorch
5. https://nlp.seas.harvard.edu/2018/04/03/attention.html
