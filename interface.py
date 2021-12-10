import numpy as np
import pandas as pd
import time

import ipywidgets as widgets
from ipywidgets import interact
from IPython.display import clear_output,display
from IPython.display import Image as disImage

from PIL import Image
import matplotlib.pyplot as plt

import konlpy
from konlpy.tag import Okt
from wordcloud import WordCloud

class Manager ():
    def __init__ (self,df):
        # 데이터 프레임 처리
        self.df = df
        self.sections = ['정치', '경제', '사회', '생활', '세계', '과학']
        
        # 워드클라우드 이미지 처리
        self.style = {"description_width": "initial"}
        icon = Image.open('./circle.png')
        mask = Image.new("RGB", icon.size, (255,255,255))
        mask.paste(icon,icon)
        self.mask = np.array(mask)
        
        # ipywidgets 처리
        self.select = widgets.Dropdown(
            options = self.sections,
            description="뉴스 기사 분야",
            disabled=False,
            continuous_update=False,
            layout={"width": "max-content"},
            readout=True,
            style=self.style)
        
        self.start = widgets.Button(description="확인")
        self.to_home_button = widgets.Button(description="뒤로 가기")
        self.back2section = widgets.Button(description="뒤로 가기")
        self.end_botton = widgets.Button(description="종료")

    def get_news(self, change): # 선택한 분야의 뉴스 목록 가져오기
        clear_output()
        
        print('해당 분야의 뉴스들을 불러오고 있습니다. 잠시만 기다려주세요')
        print('...')
        section = self.select.value
        self.my_df = self.df[self.df['분야'] == section]
        self.my_df['wc'] = self.my_df['title'] + self.my_df['contents']
        
        word_list, word_dict = self.make_wordlist()
        print('...')
        self.make_cloud(word_dict)
        
        title_list = ['모두 보기']
        title_list.extend(list(self.my_df.title.values))
        
        self.articles = widgets.Dropdown(
            options = title_list,
            description="어떤 기사의 요약문을 보시겠습니까?",
            disabled=False,
            continuous_update=False,
            layout={"width": "max-content"},
            readout=True,
            style=self.style)
        self.smz = widgets.Button(description="확인")
        
        display(self.articles)
        display(self.smz)
        display(self.to_home_button)
        display(self.end_botton)
        
        self.smz.on_click(self.summarize)
        self.to_home_button.on_click(self.go_back)
        self.end_botton.on_click(self.exit)
        
    def exit (self,change): # 프로그램 종료
        clear_output()
        
    def summarize(self, change): # 선택한 기사 요약
        clear_output()
        article = self.articles.value
        if article == '모두 보기':
            tmp_df = self.my_df
        else :
            tmp_df = self.my_df[self.my_df['title'] == article]
        
        
        for idx in range(len(tmp_df)) :
            print(f'기사 제목 : {tmp_df.title.values[idx]}\n')
            print(f'기사 요약 : {tmp_df.요약문.values[idx]}\n')
            print('------------------------------------\n')
            
        display(self.back2section)    
        display(self.end_botton)
        self.back2section.on_click(self.go_back2)
        self.end_botton.on_click(self.exit)
        
            
    def go_back(self,change) : # 뒤로가기1 : 분야 선택 창으로
        clear_output()
        self.run()
        
    def go_back2(self,change) : # 뒤로가기2 : 기사 선택 창으로
        self.get_news('ch')
        
    def make_wordlist(self): # 워드클라우드 구성을 위한 단어빈도 사전 구성
        okt = Okt()
        wlist =[]
        
        for t in self.my_df['wc'].values: # 문장 토큰화
            if t != '' :
                wlist.append(okt.pos(t))

        wdict = dict()
        for i in wlist: # 단어 딕셔너리 생성
            for j in i:
                if (j[1] == 'Noun') and (len(j[0]) > 1):
                    wdict.setdefault(j[0],0)
                    wdict[j[0]] += 1

        result = sorted(list(wdict.items()), key = lambda x :x[1],reverse = True) # 빈도수 리스트반환
        return result, wdict    
    
    def make_cloud(self,d) : # 워드클라우드 만들기
        wordcloud = WordCloud(font_path='font/malgun.ttf', background_color='white',width = 300,
                         height = 300,  mask = self.mask).generate_from_frequencies(d)
        plt.figure(figsize=(10,10)) #이미지 사이즈 지정
        plt.imshow(wordcloud, interpolation='lanczos') #interpolation: 이미지의 부드럽기 정도
        plt.axis('off') #x y 축 숫자 제거
        plt.show() 
        return 
    
    def run(self):
        print("어떤 분야의 기사들을 불러올까요?\n")
        display(self.select)
        display(self.start)
        display(self.end_botton)
        
        self.start.on_click(self.get_news)
        self.end_botton.on_click(self.exit)
