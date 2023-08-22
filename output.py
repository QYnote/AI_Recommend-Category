#데이터 입력해서 결과값 받기
print('Package 불러오기 시작')
import time
totalTime = time.time()
startTime = time.time()
import pandas as pd #DataFrame 용
import numpy as np
import joblib   #모델 불러오기용
import os
import json

from programFiles import keywordManage
from programFiles import Preprocess as pre
from programFiles import Learning

print('Package 불러오기 종료 :', time.time() - startTime)

#json파일 불러오기 : 이 파일과 같은 폴더에 있을 것
print('json 정보 가져오기')
startTime = time.time()

path = os.path.dirname(os.path.realpath(__file__)) + "\\" + 'ConstData.json'

with open(path, encoding='utf-8') as f:
    json_obj = json.load(f)

json_obj = json_obj["Run"]

print('json 정보 가져오기 종료 :', time.time() - startTime)

#엑셀파일 진행
def outputDataList():
    #적용할 파일 불러오기
    print('파일 불러오기')
    startTime = time.time()

    fileName = json_obj["runFileDirectory"] + "\\" + json_obj["runFileName"] #파일명
    dfExcel = pd.read_excel(fileName)
    df = pd.DataFrame(dfExcel, columns= [json_obj["runUseColumn"]]) #사용하는 Column만 추출

    print('파일 불러오기 종료 :', time.time() - startTime)

    ####################################################################################################
    #전처리
    print('전처리 시작')
    startTime = time.time()

    #입력할 데이터 변환
    dfInput = keywordManage.loadKeywordList()

    for row in df.loc[:, json_obj["runUseColumn"]]:    #상품명 Column 데이터만 추출[aa bb cc, dd,ee]
        keywords = []
        keywordManage.keywordExtract(insList=keywords, strValue=row)
        dfInput = pre.createInputDf(dfInput, keywords)

    print('전처리 종료 :', time.time() - startTime)
    #######################################################################################################
    #적용
    print('모델적용 시작')
    startTime = time.time()
    #최근 학습한모델 가져오기
    model = joblib.load(Learning.getRecentModel())

    #모델에 적용시긴 결과 가져오기
    result = pd.DataFrame(model.predict(dfInput))

    result['origin'] = df

    #특정확률 이하일 떄 시 빈값
    result['Rate'] = np.max(model.predict_proba(dfInput), axis=1)
    result.loc[result['Rate'] < 0.6, 0] = ''

    #Column명 바꾸기
    result.rename(columns={result.columns[0]:'카테고리 번호'}, inplace=True)
    result.rename(columns={result.columns[1]:'제목'}, inplace=True)
    result.rename(columns={result.columns[2]:'확률'}, inplace=True)

    #결과 저장
    result.to_excel(json_obj["ResultFilePath"] + "\\" + json_obj["ResultFileName"] + '.xlsx', index=False)

    print('모델적용 완료 :', time.time() - startTime)
    return result

#한문장만 진행
def outputDataOne(inputStr:str):
    #전처리
    print('전처리 시작')
    startTime = time.time()
    #입력할 데이터 변환
    dfInput = keywordManage.loadKeywordList()

    keywords = []
    keywordManage.keywordExtract(insList=keywords, strValue=inputStr)
    dfInput = pre.createInputDf(dfInput, keywords)
    
    print('전처리 종료 :', time.time() - startTime)
    #######################################################################################################
    #적용
    print('모델적용 시작')
    startTime = time.time()
    #최근 학습한모델 가져오기
    model = joblib.load(Learning.getRecentModel())

    #모델에 적용시긴 결과 가져오기
    result = pd.DataFrame(model.predict(dfInput))
    
    result['origin'] = inputStr

    #Column명 바꾸기
    result.rename(columns={result.columns[0]:'카테고리 번호'}, inplace=True)
    result.rename(columns={result.columns[1]:'제목'}, inplace=True)

    print('모델적용 종료 :', time.time() - startTime)
    return result.loc[0, result.columns[0]]

print(outputDataList())
print('최종 완료시간 :', time.time() - totalTime)

# print(outputDataOne('엑센트 2019 강화유리 내비게이션 보호 필름'))