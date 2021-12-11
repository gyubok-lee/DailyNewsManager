import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
from datetime import datetime
import os

class news_scraper ():
    """
    네이버 뉴스 기사들을 '정치', '경제', '사회', '생활', '세계', '과학' 6개 분야별로 크롤링 후
    각각 데이터 프레임으로 반환
    """
    
    def __init__ (self) :
        self.custom_header = {
        'referer' : 'https://www.naver.com/',
        "User-Agent": 
        "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36"
        }
        self.sections = ['정치', '경제', '사회', '생활', '세계', '과학']
        self.news_list = []
        
    def get_txt(self,table) : # 해당 뉴스기사의 내용을 추출
        contents = table.find(name = "div", attrs={"class" : "_article_body_contents"})

        for _ in contents.find_all("script"):
            _.replace_with('')
            
        contents = contents.get_text().replace('\n','').replace('\t', '').replace('동영상 뉴스', '')

        regex = "\[.*\]|\s-\s.*"
        contents = re.sub(regex, '',contents)
        contents = re.sub('■|▲', '',contents)
        
        return contents

    def get_href(self,soup) : # 크롤링한 soup에서 href에 해당하는 내용을 가져옴
        result = []
        div = soup.find('div', class_ = 'list_body newsflash_body')

        for dt in div.find_all('dt', class_ = 'photo') :
            result.append(dt.find("a")["href"].replace("\n", ""))
        return result


    def get_request(self,section):# 네이버 뉴스 페이지에서 분야별로 기사 크롤링
        custom_header = {
            'referer' : 'https://www.naver.com/',
            "User-Agent": 
            "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36"
        }

        url = "https://news.naver.com/main/list.nhn"

        sections = {
            "정치" : 100,
            "경제" : 101,
            "사회" : 102,
            "생활" : 103,
            "세계" : 104,
            "과학" : 105
        }

        req = requests.get(url, headers = custom_header,
                         params = {"sid1" : sections[section]})
        return req

    def start(self) : # 위 모든 함수들을 종합하여 실행
        print("----------뉴스 기사 스크랩을 실행합니다----------\n")
        list_href = []

        for section in self.sections :
            cols = ['title', 'contents','url']
            df = pd.DataFrame(columns = cols)

            req = self.get_request(section)
            soup = BeautifulSoup(req.text, "html.parser")
            

            list_href = self.get_href(soup)

            for href in list_href :
                href_req = requests.get(href, headers = self.custom_header)
                href_soup = BeautifulSoup(href_req.text, "html.parser")
                url = href_soup.select_one('meta[property="og:url"]')['content']
                try:
                    table = href_soup.find(name = 'td', attrs = {"class":"content"})
                    title = table.find(name = 'div', attrs = {"class":"article_info"}).find('h3').get_text()
                    contents = self.get_txt(table)
                    row = pd.Series([title,contents,url],index = df.columns)
                    df = df.append(row, ignore_index = True)
                except :
                    continue
                
            self.news_list.append(df)
            print(f'{section} 뉴스 기사 크롤링 완료')
        return self.news_list
