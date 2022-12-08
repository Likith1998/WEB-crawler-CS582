from html.parser import HTMLParser
import pickle
import re
import string
from nltk.stem import PorterStemmer
import os
from bs4 import BeautifulSoup
from bs4.element import Comment
from nltk.corpus import stopwords


l=[]
inverted_index={}
doc_tokens={}
for filename in os.listdir(".\CrawledData"):
    l.append(filename)


def text_html(body):
    v=[]
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    for i in texts:
        if i.parent.name not in ['style', 'script', 'head', 'meta', '[document]'] and isinstance(i,Comment)==False:
            v.append(i)
    
    return u" ".join(t.strip() for t in v)
    
def extract():
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    for filename in l:
        d = open(".\CrawledData"+"\\"+filename,"r",encoding="utf-8") 
        html=d.read()
        text=text_html(html)
        
        words = " ".join(re.findall("[a-zA-Z0-9]+", text))
        text = ''.join(i for i in words if not i.isdigit())         
        text=text.lower()   
        words = text.split(" ")
        f= [ps.stem(x) for x in words if ps.stem(x) not in stop_words and len(ps.stem(x))>2]              #Removes stop words and performs stemming
        for token in f:
            inverted_index.setdefault(token,{})[filename]= inverted_index.setdefault(token, {}).get(filename, 0) + 1
            
        doc_tokens[filename]=f

        



def sotre_inverted_index():
    with open('inverted_index.pickle', 'wb') as f:
        pickle.dump(inverted_index, f)

def store_doc_tokens():
    with open('docs_tokens.pickle', 'wb') as f:
        pickle.dump(doc_tokens, f)

            
extract()
sotre_inverted_index()
store_doc_tokens()
