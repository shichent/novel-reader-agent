from sklearn.utils import murmurhash3_32
import matplotlib.pyplot as plt
import pandas as pd
import random

def MinHash(A,K,gram=1):
    ans = []
    for seed in range(K):
        min_hash = float('inf')
        for j in range(len(A)-gram+1):
            h_val = murmurhash3_32(A[j:j+gram], seed=seed+42) % (2**20)
            if h_val < min_hash:
                min_hash = h_val
        ans.append(min_hash)
    return ans

class HashTable():
    def __init__(self,K,L,B,R):
        self.K = K  # Number of hash functions per table
        self.L = L  # Number of hash tables
        self.R = R  # Range for hash functions， i.e. number of buckets per table
        self.B = B
        random.seed(42)
        self.a = random.sample(range(1,1000),K)
        self.c = random.randint(0,1000)
        self.tables = [dict() for _ in range(L)]

    def insert(self,hashcodes,id):
        for i in range(self.L):
            keys = hashcodes[i*self.K:(i+1)*self.K]
            bucket = sum([self.a[j]*keys[j] for j in range(self.K)]) % self.R
            bucket = bucket%self.B
            if bucket not in self.tables[i]:
                self.tables[i][bucket] = []
            self.tables[i][bucket].append(id)

    def lookup(self,hashcodes):
        ans = []
        for i in range(self.L):
            keys = hashcodes[i*self.K:(i+1)*self.K]
            bucket = sum([self.a[j]*keys[j] for j in range(self.K)]) % self.R
            bucket = bucket%self.B
            if bucket in self.tables[i]:
                ans += self.tables[i][bucket]
        return list(set(ans))

def jaccard_similarity(s1,s2,gram = 3):
    set1 = set(s1[i:i+gram] for i in range(len(s1)-gram+1))
    set2 = set(s2[i:i+gram] for i in range(len(s2)-gram+1))
    return len(set1.intersection(set2)) / len(set1.union(set2))

def plot_histogram(data, title, xlabel):
    series = pd.Series(data)
    ax = series.hist(bins=20)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig = ax.get_figure()
    fig.savefig(title+'.png')

def pairwise_jaccard(urls):
    n = len(urls)
    jaccard_matrix = [[0]*n for _ in range(n)]
    for i in range(n):
        jaccard_matrix[i][i] = 1.0
        for j in range(i+1, n):
            sim = jaccard_similarity(urls[i], urls[j])
            jaccard_matrix[i][j] = sim
            jaccard_matrix[j][i] = sim
    return jaccard_matrix

if __name__ == "__main__":
    S1 = "The mission statement of the WCSCC and area employers recognize the importance of good attendance on the job. Any student whose absences exceed 18 days is jeopardizing their opportunity for advanced placement as well as hindering his/her likelihood for successfully completing their program."
    S2 = "The WCSCC’s mission statement and surrounding employers recognize the importance of great attendance. Any student who is absent more than 18 days will loose the opportunity for successfully completing their trade program."
    K,L,B,R = 2,50,64,2**20
    B,R = 64,2**20
    hashcodes = MinHash(S1,K*L)
    hashtable = HashTable(K,L,B,R)
    hashtable.insert(hashcodes,1)
    hashcodes = MinHash(S2,K*L)
    start = pd.Timestamp.now()
    candidates = hashtable.lookup(hashcodes)
    end = pd.Timestamp.now()
    query_times = ((end - start).total_seconds())
    sim = jaccard_similarity(S1,S2)