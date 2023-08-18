import pandas as pd #DataFrame 용
import datetime
import csv #Input Keyword 관리용
import os

import time
from konlpy.tag import Komoran
komoran = Komoran()
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(
    min_df = 2  #최소 사용 단어량
)

#키워드리스트 저장
def saveKeyword(keywords:list):
    fileName = os.path.dirname(os.path.realpath(__file__)) + "\\" + 'List_Category' + '.csv'
    path = fileName #경로 + 파일명

    with open(path, 'w', encoding='utf8') as file:
        write = csv.writer(file)
        write.writerow(keywords)
        write.writerow(['최근 변경날짜:' + str(datetime.date.today())])

#키워드리스트 불러오기
def loadKeywordList():
    fileName = os.path.dirname(os.path.realpath(__file__)) + "\\" + 'List_Category' + '.csv'
    path = fileName #경로 + 파일명
    
    dfKeywordFrame = pd.read_csv(path, encoding='utf8')
    dfKeywordFrame.drop(index=0, axis=0, inplace=True) #날짜 입력한 Row 삭제
    dfKeywordFrame = dfKeywordFrame.reset_index(drop=True)

    return dfKeywordFrame

#키워드 추출
def keywordExtract(insList:list, strValue:str, outStr:list=None):

    UseTag = ["NNG", "NNP", "NP"]   #입력할 구분()
    TagList = []

    title = komoran.pos(strValue)

    #Title의 키워드들 점검
    for keyword in title:
        if(keyword[1] in UseTag):    #입력할 Tag에 속하면 입력
            if(outStr is not None and keyword[0] in outStr):
                continue
            insList.append(keyword[0])

        # elif(keyword[1] == "SL"):#외국어 처리?
        #     print("외국어 :", keyword[0])
        # else:   #개발용
        #     if(keyword[1] not in TagList):
        #         TagList.append(keyword[1])

def keywordVectorizer(keywords:list):
    #Vector화
    vectorizer.fit_transform(keywords)      #상품명에있는 키워드들 리스트화

    return vectorizer.get_feature_names_out()