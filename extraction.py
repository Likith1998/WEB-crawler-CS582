import math
from collections import Counter
import operator
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle

N=3000      
idf={}      
doc_length_1={}

def tf_idf(word, doc):
    inverted_index_tfidf[word][doc] = inverted_index_tfidf[word][doc] * idf[word]
    return inverted_index_tfidf[word][doc]

def compute_idf(inverted_index_tfidf):
    df={}
    idf={}
    for key in inverted_index_tfidf.keys():
        df[key] = len(inverted_index_tfidf[key].keys())
        idf[key] = math.log(N / df[key], 2)

    return idf

def compute_all_tf_idf():
    for word in inverted_index_tfidf:
        for doc_key in inverted_indextfidf[word]:
            inverted_index_tfidf[word][doc_key] = inverted_index_tfidf[word][doc]*idf[word]
    return inverted_index_tfidf

def compute_lengths(docs_tokens):
    for code in range(N+1):
        doc_length_1[str(code)] = compute_doc_length(str(code), docs_tokens[str(code)])
    return doc_length_1

def compute_doc_length(code, tokens):
    words_accounted_for = []
    length = 0
    for token in tokens:
        if token not in words_accounted_for:
            length += tf_idf(token, code) ** 2
            words_accounted_for.append(token)
    return math.sqrt(length)



def cosine_similarities(query,doc_length):
    length = 0
    cnt = Counter()
    similarity = {}
    query_length=0
    for w in query:
        cnt[w] += 1
    for w in cnt.keys():
        length += (cnt[w]*idf.get(w, 0)) ** 2
        
    query_length=math.sqrt(length)
    
    for word in query:
        wq = idf.get(word, 0)
        if wq != 0:
            for doc in inverted_index_tfidf[word].keys():
                similarity[doc] = similarity.get(doc, 0) + inverted_index_tfidf[word][doc] * wq
      
    for doc in similarity.keys():
        similarity[doc] = similarity[doc] / doc_length[doc] / query_length
             
    return similarity

def retrieve_most_relevant(query_tokens,doc_length):
    similarity=cosine_similarities(query_tokens,doc_length)
    order=sorted(similarity.items(), key=operator.itemgetter(1), reverse=True)
    return order


def add_page_rank_scores_and_reorder(best_ranked, page_ranks):
    best_dict = dict(best_ranked)
    for doc_code in best_dict:
        best_dict[doc_code] = best_dict[doc_code] + page_ranks[doc_code] * PAGE_RANK_MULTIPLIER

    return page_ranks(best_dict)

def print_output(top,n,iter):
    for i in range(n,n+iter):
        l=top[i][0]
        if(links[int(l)]==None):
            exit()
        elif(links[int(l)]!=None):
            print(links[int(l)])
        else:
            break

    
with open('inverted_index.pickle', 'rb') as f:
    inverted_index_tfidf=pickle.load(f)

with open('docs_tokens.pickle', 'rb') as f:
    docs=pickle.load(f)

with open('url.pickle', 'rb') as f:
    links=pickle.load(f)                    


idf=compute_idf(inverted_index_tfidf)
doc_length=compute_lengths(docs)


stop_words = set(stopwords.words('english'))
query=str(input("Enter your search:"))
ps = PorterStemmer()

for c in string.punctuation:
    query=query.replace(c,"")           


query_temp_string=""
for i in query:
    if not i.isdigit():
        query_temp_string+=i

query=query_temp_string.lower()


f = word_tokenize(query)

f_new=[]
for x in f:
    if x not in stop_words:
        f_new.append(ps.stem(x))


unique=set(f)                                                      
top=retrieve_most_relevant(unique,doc_length)                          
n=0
if not len(top):
    print("no pages scraped for the following input, please try again")
if len(top)<10:
    print_output(top,n,len(top))
else:
    print_output(top,n,10)
    n=n+10
    u=input("Do you want more pages? Enter yes/no:")
    counter_pages=10
    while(u=='yes'):
        if len(top)-(counter_pages+10)<0:
            print_output(top,n,len(top)-counter_pages)    
            print("parsing of pages done")
            break
        else:
            print_output(top,n,10)
            n=n+10
            counter_pages+=10
            u=input("Do you want more pages? Enter yes/no:")

