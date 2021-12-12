
import numpy as np
import pandas as pd
import time
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences

from konlpy.tag import Okt, Kkma
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize
from tqdm import tqdm
tqdm.pandas()
np.random.seed(seed=0)

class news_summarizing():
    """
    데이터를 요약가능한 데이터와 그렇지 않은 데이터 둘로 나눔
    """
    
    def __init__(self, news_list):
        df = news_list[0]
        df['분야'] = '정치'
        secs = ['정치','경제', '사회', '생활', '세계', '과학']
        for i in range(1,len(news_list)):
            now = news_list[i]
            now['분야'] = secs[i]
            df = pd.concat([df,now], axis = 0)
            
        self.df = df 
        self.df1 = df[df['contents'].str.len() < 300] # 300자 미만의 기사들은 요약없이 내용을 그대로
        self.df2 = df[df['contents'].str.len() >= 300]
    
    def text2sentence (self,text) : # 기사 한 문단을 하나의 문장 리스트로
        kkma = Kkma()
        sentence = kkma.sentences(text)
        for i in range(len(sentence)):
            if len(sentence[i]) <= 10:
                sentence[i-1] += (' '+sentence[i])
                sentence[i] = ''
        return sentence
    
    def give_df(self):
        time.sleep(1.5)
        print('----------뉴스 데이터를 처리 중입니다----------')
        self.df2['s1'] = self.df2['contents'].progress_apply(lambda x: self.text2sentence(x))
        return self.df1,self.df2
    

class smz() :
    """
    추출적 요약을 통해 뉴스 기사를 3문장으로 요약.
    TextRank 기법을 통해 요약모델을 완성
    """
    
    tfd = TfidfVectorizer()
    cnt_vec = CountVectorizer()
    

    def cleansing(text): # 명사들만 추출
        okt =Okt()
        stopwords = ['머니투데이', '연합뉴스', '데일리', '동아일보', '중앙일보',
                    '조선일보', '기자']

        nouns = []
        for sentence in text :
            if sentence is not '': # 1자 이상이면서 stopword가 아닌 명사들 추출
                nouns.append(' '.join([noun for noun in okt.nouns(str(sentence))
                                       if noun not in stopwords and len(noun) > 1]))
        return nouns


    def mk_sentGraph(x): # 문장 간 voting graph 생성
        tfd_mat = smz.tfd.fit_transform(x).toarray()
        gs = np.dot(tfd_mat, tfd_mat.T)
        return gs

    def mk_wordsGraph(x):# 단어 간 voting graph 생성
        cnt_vec_mat = normalize(smz.cnt_vec.fit_transform(x).toarray().astype(float), axis =0)
        voca = smz.cnt_vec.vocabulary_
        return np.dot(cnt_vec_mat.T, cnt_vec_mat), {voca[i] : i for i in voca}

    def get_ranks(graph, d = 0.85) : # text rank 구하기
        A = graph
        matrix_size = A.shape[0]
        
        for i in range(matrix_size):
            A[i,i] = 0
            link_sum = np.sum(A[:,i])
            if link_sum != 0:
                A[:,i] /= link_sum
            A[:,i] *= -d
            A[i,i] = 1
            
        B = (1-d) * np.ones((matrix_size,1))
        ranks = np.linalg.solve(A,B) # 선형방정식 solve
        
        return {idx: r[0] for idx, r in enumerate(ranks)}

    def summarize(sentences,ranked): # 상위 sunt_num개의 문장을 추출
        sent_num =3
        summary = []
        index = []
        for idx in ranked[:sent_num] :
            index.append(idx)
        index.sort()

        for idx in index:
            summary.append(sentences[idx])
        return '\n'.join(summary)
            
    def run(x) : 
        cleaned = smz.cleansing(x)

        try :
            # 그래프 만들기
            sent_graph = smz.mk_sentGraph(cleaned)
            words_graph, idx2word = smz.mk_wordsGraph(cleaned)

            # 순위 매기기
            sent_rank_idx = smz.get_ranks(sent_graph)
            word_rank_idx = smz.get_ranks(words_graph)

            # 정렬
            sorted_sent_rank_idx = sorted(sent_rank_idx,
                                  key = lambda x : sent_rank_idx[x],
                                          reverse = True)
            sorted_word_rank_idx = sorted(word_rank_idx,
                                  key = lambda x : word_rank_idx[x],
                                          reverse = True)

            # 순서대로 문장 추출
            result = smz.summarize(x,sorted_sent_rank_idx)
            return result
        except :
            print(f'요약에 실패하였습니다. 기사 내용 : {x[0][:50]} ......')
            return ''
        
    def concatDF(df1, df2):
        print('----------뉴스 기사 요약문이 완성되었습니다 ----------')
        df1['요약문'] = df1['contents']
        df_final = pd.concat([df1,df2], axis =0).sort_values(by=['분야'])
        df_final.loc[df_final.요약문 == '', '요약문'] = df_final.loc[df_final.요약문 == '', 'title']
        return df_final.drop(['s1'], axis=1).reset_index(drop = True)
