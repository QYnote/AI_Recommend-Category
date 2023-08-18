import pandas as pd #DataFrame 용
import datetime
# import tkinter.messagebox as msgbox
import joblib   #모델 저장용
import json
import os

# import defGlobal
from programFiles import Preprocess as pre
from programFiles import keywordManage
from programFiles import Learning

#Excel Data 가져오기

#from konlpy.tag import Okt #한글 텍스트 처리하는 Package
#okt = Okt()

print('json파일 가져오기')
#json파일 불러오기 : 이 파일과 같은 폴더에 있을 것
path = os.path.dirname(os.path.realpath(__file__)) + "\\" + 'ConstData.json'

with open(path, encoding='utf-8') as f:
    json_obj = json.load(f)

json_obj = json_obj["Learning"]
print('json파일 가져오기 종료')

#학습할 엑셀 파일 불러오기
#테스트용 : '카테고리명', 실사용: '마켓 카테고리번호'
useColumn = json_obj["learningUseColumn"] #사용할 Column명
path = json_obj["learningFileDirectory"] + '\\' + json_obj["learningFileName"]

dfExcel = pd.read_excel(path)
dfExcel = dfExcel.drop([0]) #첫번째 설명Row 삭제
df = pd.DataFrame(dfExcel, columns= useColumn) #사용하는 Column만 추출
df = df.reset_index(drop=True)

###############################################################################################
#1차가공
print('1차가공 시작')

#Output Data 가공 : 누나꺼 전용
#쿠팡의 카테고리코드번호로 나오도록 출력
#줄바꿈(char(10))으로 split -> '*'로 split -> [1]번째 Column값으로 변경
#쿠팡이 항상 첫번재 행의 쿠팡*1234로 이루어져있다고함
# if(useColumn[0] == '마켓 카테고리번호'):
#     for idx in range(0, len(df)):
#         if(type(df.loc[idx, useColumn[0]]) == str):
#             if('쿠팡*' in df.loc[idx, useColumn[0]]):
#                 df.loc[idx, useColumn[0]] = df.loc[idx, useColumn[0]].splitlines(keepends=False)[0].split('*')[1]
#         else:
#             df.drop([idx], inplace=True)
            
#     df = df.reset_index(drop=True)

# category_dic = {}
#     for j in range(len(category_list)):
#         category_list[j]= category_list[j].split("*")
#         if category_list[j] == ['']:
#             pass
#         else:
#             category_dic[category_list[j][0]] = category_list[j][1]

#Input용 키워드 리스트 추출 및 저장
keywords = []
for row in df.loc[:, useColumn[1]]:
    keywordManage.keywordExtract(insList=keywords, strValue=row, outStr=json_obj["ExceptionWord"])

keywords = keywordManage.keywordVectorizer(keywords=keywords)
keywordManage.saveKeyword(keywords)

print('1차가공 종료')
###############################################################################################
#2차가공
print('2차가공 시작')
#전처리
#Input용 키워드 리스트 호출
dfInput = keywordManage.loadKeywordList()
dfOutput = df.loc[:, useColumn[0]]

#Input DataFrame 제작
for row in df.loc[:, useColumn[1]]:    #상품명 Column 데이터만 추출[aa bb cc, dd,ee]
    keywords = []
    keywordManage.keywordExtract(insList=keywords, strValue=row)
    dfInput = pre.createInputDf(dfInput, keywords)

#Input 데이터 증가
minCnt = json_obj["MinLearningDataCnt"]

if(dfInput.shape[0] != 0 and dfOutput.shape[0] != 0):
    while(dfInput.shape[0] < minCnt and dfOutput.shape[0] < minCnt):
        # pandas 2.0.0 이상
        dfInput = pd.concat([dfInput, dfInput])
        dfOutput = pd.concat([dfOutput, dfOutput])

        # #pandas 2.0.0 ver 미만
        # dfInput = dfInput.append(dfInput)
        # dfOutput = dfOutput.append(dfOutput)

print('2차가공 종료')

###############################################################################################
#데이터 학습
print('학습시작')
model, result = Learning.learning_KNN(dfInput, dfOutput)

#모델저장 경로 없으면 생성
path = os.path.dirname(os.path.realpath(__file__)) + "\\programFiles\\models"
try:
    if not os.path.exists(path):
        os.makedirs(path)
except OSError:
    print("Error: Failed to create the directory.")


#모델 저장
#모델명 : 모델명_날짜_시분_정확도
modelName = "KNN" + "_" + str(datetime.datetime.now().strftime('%Y%m%d_%H%M')) + "_" + '{:.2f}'.format(result)
path = path + "\\" + modelName + ".pkl"

joblib.dump(model, path)

print('학습종료')