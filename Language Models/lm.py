##
import os.path
import sys
import random
from operator import itemgetter
import numpy as np
from collections import defaultdict


#this funtion takes a file name and creates a list of sentence corpus
def readFileToCorpus(f):
    """ Reading the text file f which contains one sentence per line.
    """
    if os.path.isfile(f):
        file = open(f, "r") # opening the input file in read onli mode
        i = 0               # counter to keep track of the sentence numbers
        corpus = []         # this will become a list of sentences
        print("Reading file ", f)
        
        for line in file:
            i += 1
            sentence = line.split() # spliting the line into a list of words
            corpus.append(sentence) # append this lis as an element to the list of sentences
            
            if i % 1000 == 0:
                sys.stderr.write("Reading sentence " + str(i) + "\n")
            #endif
        #endfor
        return corpus
    else:
        #throwing an exception here
        print("Error: corpus file ", f, " does not exist")
        sys.exit() # exiting the script
    #endif
#enddef


# This function Preprocess the corpus
def preprocess(corpus):
    #find ing all the rare words
    freqDict = defaultdict(int)
    for sen in corpus:
	    for word in sen:
	       freqDict[word] += 1
	    #endfor
    #endfor

    #replacing rare words with unk
    for sen in corpus:
        for i in range(0, len(sen)):
            word = sen[i]
            if freqDict[word] < 2:
                sen[i] = UNK
	        #endif
	    #endfor
    #endfor

    #adding the starting and ending (<s> and </s>) tokens in each sentence
    for sen in corpus:
        sen.insert(0, start)
        sen.append(end)
    #endfor
    
    return corpus
#enddef

def preprocessTest(vocab, corpus):
    #replacing test words that were unseen in the training with unk
    for sen in corpus:
        for i in range(0, len(sen)):
            word = sen[i]
            if word not in vocab:
                sen[i] = UNK
	        #endif
	    #endfor
    #endfor
    
     #adding the starting and ending (<s> and </s>) tokens in each sentence
    for sen in corpus:
        sen.insert(0, start)
        sen.append(end)
    #endfor

    return corpus
#enddef


UNK = "UNK"     # Unknown word token
start = "<s>"   # Start of sentence token
end = "</s>"    # end of senetnce token


# Language models and data structures


# Parent class for the language models
class LanguageModel:
   
    def __init__(self, corpus):
        pass
    #enddef


    def generateSentence(self):
        return "mary had a little lamb ."
    #emddef

    def getSentenceProbability(self, sen):
        return 0.0
    #enddef

  
    def getCorpusPerplexity(self, corpus):
        return 0.0
    #enddef

   #this function sotres senetnces to files
    def generateSentencesToFile(self, numberOfSentences, filename):
        filePointer = open(filename, 'w+')
        for i in range(0,numberOfSentences):
            sen = self.generateSentence()
            prob = self.getSentenceProbability(sen)

            stringGenerated = str(prob) + " " + " ".join(sen) 
            print(stringGenerated, end="\n", file=filePointer)
        #endfor
    #enddef
#endclass

# Unigram language model
class UnigramModel(LanguageModel):
    
    def __init__(self, corpus):
        self.distribution=UnigramDist(corpus)
    #endddef
    
    def generateSentence(self):
        sentence=[start]
        
        while True:
            word=self.distribution.draw()
            if word==end:
                sentence.append(end)
                break
            #endif
            sentence.append(word)
        #endwhile
        return sentence
    #enddef
    
    def getSentenceProbability(self, sen):
        log_prob = 0.0
        for word in sen:
            if word in {start, end}:
                continue
            p = self.distribution.prob(word)
            if p > 0.0:
                log_prob += np.log2(p)
            else:
                return float('inf')
            #endif
        #endfor
        return 2**log_prob
    #enddef

    
    def getCorpusPerplexity(self, corpus):
        total_log_prob = 0.0
        word_count = 0

        for sen in corpus:
            for word in sen[1:-1]: 
                prob = self.distribution.prob(word)
                if prob > 0.0:
                    total_log_prob += np.log2(prob)
                else:
                    return float('inf')
                word_count += 1
            #endfor
        #endfor

        avg_log_prob = total_log_prob / word_count
        perplexity = 2 ** (-avg_log_prob)

        return perplexity
    #enddef     
#endclass

#Smoothed unigram language model (useing laplace  add-1 for smoothing)
class SmoothedUnigramModel(LanguageModel):
    def __init__(self, corpus):
        self.distribution=UnigramDist(corpus)
    #endddef
    
    def generateSentence(self):
        sentence=[start]
        
        while True:
            word=self.distribution.draw()
            if word==end:
                sentence.append(end)
                break
            #endif
            sentence.append(word)
        #endwhile
        return sentence
    #enddef
    
    def getSentenceProbability(self, sen):
        log_prob = 0.0
        for word in sen:
            if word in {start, end}:
                continue
            p = self.distribution.laplace_prob(word)
            if p > 0.0:
                log_prob += np.log2(p)
            else:
                return float('inf')
            #endif
        #endfor
        return 2**log_prob
    #enddef

    
    def getCorpusPerplexity(self, corpus):
        total_log_prob = 0.0
        word_count = 0

        for sen in corpus:
            for word in sen[1:-1]: 
                prob = self.distribution.laplace_prob(word)
                if prob > 0.0:
                    total_log_prob += np.log2(prob)
                else:
                    return float('inf')
                word_count += 1
            #endfor
        #endfor

        avg_log_prob = total_log_prob / word_count
        perplexity = 2 ** (-avg_log_prob)

        return perplexity
    #enddef
#endclass



# Unsmoothed bigram language model
class BigramModel(LanguageModel):
    def __init__(self, corpus):
        self.distribution=BigramDist(corpus)
    #endddef
    
    def generateSentence(self):
        sentence=[start]
       
        while True:
            previous=sentence[-1]
            word=self.distribution.draw(previous)
            if word==end:
                sentence.append(word)
                break
            #endif
            sentence.append(word)
        #endwhile
        return sentence
    #enddef
    
    def getSentenceProbability(self, sen):
        log_prob=0.0
        
        for i in range(1,len(sen)):
            current=sen[i]
            previous=sen[i-1]
            p=self.distribution.probability(previous,current)
            
            if p > 0.0:
                log_prob+=np.log2(p)
            else:
                return float('inf') 
        #endfor
        return 2**log_prob
    #enddef
    

    def getCorpusPerplexity(self, corpus):
        total_log_prob = 0.0
        total_words = 0
        for sen in corpus:
            for i in range(1, len(sen)):
                prev = sen[i-1]
                word = sen[i]
                p = self.distribution.probability(prev, word)
                if p > 0.0:
                    total_log_prob += np.log2(p)
                else:
                    return float('inf')
                total_words += 1
            #endfor
        return 2 ** (-total_log_prob / total_words)
    #enddef
        
#endclass



# Smoothed bigram language model (using linear interpolation for smoothing, setting lambda1 = lambda2 = 0.5)
class SmoothedBigramModelKN(LanguageModel):
    def __init__(self, corpus):
        self.bi_distribution=BigramDist(corpus)
        self.uni_distribution=UnigramDist(corpus)
        self.lambda1=0.5
        self.lambda2=0.5
        
    #endddef
    
    def generateSentence(self):
        sentence=[start]
       
        while True:
            previous=sentence[-1]
            word=self.bi_distribution.draw(previous)
            if word==end:
                sentence.append(word)
                break
            #endif
            sentence.append(word)
        #endwhile
        return sentence
    #enddef
    
    def getSentenceProbability(self, sen):
        log_prob=0.0
        
        for i in range(1,len(sen)):
            current=sen[i]
            previous=sen[i-1]
            p_bigram=self.bi_distribution.probability(previous,current)
            p_unigram=self.uni_distribution.prob(current)
            
            probb=p_bigram*self.lambda1+p_unigram*self.lambda2
            
            if probb > 0.0:
                log_prob+=np.log2(probb)
            else:
                return float('inf')  
        #endfor
        return 2**log_prob
    #enddef
    

    def getCorpusPerplexity(self, corpus):
        total_log_prob = 0.0
        total_words = 0

        for sen in corpus:
            for i in range(1, len(sen)):
                prev = sen[i-1]
                word = sen[i]

                p_bigram = self.bi_distribution.probability(prev, word)
                p_unigram = self.uni_distribution.prob(word)

                p = self.lambda1 * p_bigram + self.lambda2 * p_unigram
                if p > 0.0:
                    total_log_prob += np.log2(p)
                else:
                    return float('inf')
                total_words += 1
        return 2 ** (-total_log_prob / total_words)
#endclass



class UnigramDist:
    def __init__(self, corpus):
        self.counts = defaultdict(float)
        self.total = 0.0
        self.train(corpus)
        self.V = len(self.counts)
    #endddef

    # Add observed counts from corpus to the distribution
    def train(self, corpus):
        for sen in corpus:
            for word in sen:
                self.counts[word] += 1.0
                self.total += 1.0
            #endfor
        #endfor
    #enddef

    # Returns the probability of word in the distribution
    def prob(self, word):
        return self.counts[word]/self.total
    #enddef
    
    def laplace_prob(self, word):
        return (self.counts[word] + 1.0) / (self.total + self.V)
    #enddef


    # Generates a single random word according to the distribution
    def draw(self):
        rand = random.random()
        for word in self.counts.keys():
            rand -= self.prob(word)
            if rand <= 0.0:
                return word
	    #endif
	#endfor
    #enddef
#endclass


class BigramDist:
    
    def __init__(self,corpus):
        self.bi_gram_counts=defaultdict(float)
        self.uni_gram_counts=defaultdict(float)
        self.total=0
        self.train_model(corpus)
        self.vocabulary=len(self.uni_gram_counts)
    #enddef
    
    def train_model(self,corpus):
        
        for sen in corpus:
            for i in range(1,len(sen)):
                current=sen[i]
                previous=sen[i-1]
                self.bi_gram_counts[(previous,current)]+=1.0
                self.uni_gram_counts[previous]+=1.0
                self.total+=1.0
            #endfor
        #endfor
    #enddef
    
    def probability(self,previous,current):
        if self.uni_gram_counts[previous]==0:
            return 0
        else:
            return self.bi_gram_counts[(previous,current)] / self.uni_gram_counts[previous]
    #enddef
    
    def draw(self,previous):
        
        rand=random.random()
        candidates = [w for (prev, w) in self.bi_gram_counts if prev == previous]
        if not candidates:
          return end 
    
        for word in candidates:
            rand-=self.probability(previous,word)
            if rand<=0.0:
                return word
            
        return candidates[-1]
            
#-------------------------------------------
# The main routine
#-------------------------------------------
if __name__ == "__main__":
    
    #readind train corpus
    trainCorpus = readFileToCorpus('train.txt')
    trainCorpus = preprocess(trainCorpus)
    
    #reading files for test corpus
    posTestCorpus = readFileToCorpus('pos_test.txt')
    negTestCorpus = readFileToCorpus('neg_test.txt')
    
    
    vocab = []
    #creating a vocabulary(collection of word types) for the train corpus
    for sen in trainCorpus:
        for word in sen:
            vocab.append(word)
    vocab=set(vocab)
    print('\n\n------------------------Following is my Vocabulary-------------------------------\n\n')
    print(f'The Vocabulary size is : {len(vocab)}')
    
    posTestCorpus = preprocessTest(vocab, posTestCorpus)
    negTestCorpus = preprocessTest(vocab, negTestCorpus)
    
    
    unigram=UnigramModel(trainCorpus)
    unigramSmoothed=SmoothedUnigramModel(trainCorpus)
    bigram=BigramModel(trainCorpus)
    bigramSmoothed=SmoothedBigramModelKN(trainCorpus)
    
   
    unigram.generateSentencesToFile(20,"unigram.txt")
    unigramSmoothed.generateSentencesToFile(20,"unigram_smoothed.txt")
    bigram.generateSentencesToFile(20,"bigram.txt")
    bigramSmoothed.generateSentencesToFile(20,"bigram_smoothed.txt")
    
    



    unigram_pos_perplexity = unigram.getCorpusPerplexity(posTestCorpus)
    unigram_neg_perplexity = unigram.getCorpusPerplexity(negTestCorpus)
    print("Unigram Model Perplexity:")
    print("POS Test Corpus:", unigram_pos_perplexity)
    print("NEG Test Corpus:", unigram_neg_perplexity)




    smoothed_unigram_pos = unigramSmoothed.getCorpusPerplexity(posTestCorpus)
    smoothed_unigram_neg = unigramSmoothed.getCorpusPerplexity(negTestCorpus)
    print("\nSmoothed Unigram Model Perplexity:")
    print("POS Test Corpus:", smoothed_unigram_pos)
    print("NEG Test Corpus:", smoothed_unigram_neg)


    bigram_pos = bigram.getCorpusPerplexity(posTestCorpus)
    bigram_neg = bigram.getCorpusPerplexity(negTestCorpus)
    print("\nBigram Model Perplexity:")
    print("POS Test Corpus:", bigram_pos)
    print("NEG Test Corpus:", bigram_neg)



    smoothed_bigram_pos = bigramSmoothed.getCorpusPerplexity(posTestCorpus)
    smoothed_bigram_neg = bigramSmoothed.getCorpusPerplexity(negTestCorpus)
    print("\nSmoothed Bigram Model Perplexity:")
    print("POS Test Corpus:", smoothed_bigram_pos)
    print("NEG Test Corpus:", smoothed_bigram_neg)