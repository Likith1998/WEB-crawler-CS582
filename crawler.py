import requests
from bs4 import BeautifulSoup
from collections import deque
import os
import urllib.request
import pickle
import ssl

def crawler():
    q=deque()
    ssl._create_default_https_context = ssl._create_unverified_context
    url="https://www.cs.uic.edu/"
    q.append(url)
    url_list=[]
    domain="uic.edu"
    url_list.append(domain)
    exclude_content = ['.avi','.ppt','.gz','.zip','.tar','.tgz','.docx','.ico', '.css', '.js','.jpg', '.jpeg', '.png', '.gif', '.pdf', '.doc', '.JPG', '.mp4', '.svg']
    pageNo=0
    url={}
    out_links=[]
    while(len(q)!=0):
        try:
            print("Scraping page: {}".format(pageNo))
            link=q.popleft()
            r=requests.get(link, verify=False)                
            if(r.status_code == 200):
                url[pageNo]=link
                filename = "./CrawledData/"+str(pageNo)
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, "w",encoding="utf-8") as f:
                    f.write(r.text)
                              
                soup = BeautifulSoup(r.text,'lxml')
                
                tags=soup.find_all('a')
                    
                for tag in tags:
                    l=tag.get('href')
                    if l is not None and l.startswith("http") and l not in url_list and domain in l and \
                    not any(word in l for word in exclude_content):
                        url_list.append(l)
                        q.append(l)
                            
                if(len(url) > 3000):
                    break

                    
                pageNo=pageNo+1

        except Exception as e:
            print(e)
            print("Connection failed for", link)
            continue
        
    with open('url.pickle', 'wb') as f:
        pickle.dump(url, f)
        
                
crawler()
