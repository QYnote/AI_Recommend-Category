print('Package 호출 중')
import pandas as pd #DataFrame 용
import json
import os
import tkinter as tk
from tkinter import filedialog
import time

from programFiles import Model
print('Package 호출 완료')

###############################################################################################
print('json파일 가져오기')
#json파일 불러오기 : 이 파일과 같은 폴더에 있을 것
path = os.path.dirname(os.path.realpath(__file__)) + "\\" + 'ConstData.json'

with open(path, encoding='utf-8') as f:
    json_obj = json.load(f)

json_obj = json_obj["Learning"]
print('json파일 가져오기 종료')

###############################################################################################
#학습할 엑셀 파일 불러오기
#테스트용 : '카테고리명', 실사용: '마켓 카테고리번호'
print('Excel 파일 가져오기')
root = tk.Tk()
root.withdraw() #Python 기본 윈도우 숨기기
openFilePath = filedialog.askopenfilename(
    filetypes=(("xlsx File", "*.xlsx"), ("xls File", "*.xls")),
    title='적용파일 불러오기'
)
# openFilePath = "D:/00.Storage/Study/Programing/Project/AI_Recommend-Category/OC_category_review_2023-08-18_.xlsx"
if(openFilePath == ''):
    exit()

useColumn = json_obj["learningUseColumn"] #사용할 Column명

dfExcel = pd.read_excel(openFilePath)
dfExcel = dfExcel.drop([0]) #첫번째 설명Row 삭제
df = pd.DataFrame(dfExcel, columns= useColumn) #사용하는 Column만 추출
df = df.reset_index(drop=True)

print('Excel 파일 가져오기 종료')

###############################################################################################
#모델 전처리
print('전처리 시작')
pre = Model.PreProcess()
dataList = []

#키워드 추출
for row in df.loc[:, useColumn[1]]:
    dataList.append(pre.KeywordCheck(row, ExceptionWords=json_obj["ExceptionWords"]))

TrainData = pre.Vectorise_Learn(dataList)   #str -> float화
LabelData = pre.Platform_Modify(df.loc[:, useColumn[0]])    #플랫폼별 분리

#결측치 제거
LearningData = pd.concat([TrainData, LabelData], axis=1).dropna()

while(LearningData.shape[0] < json_obj["MinLearningDataCnt"]):
    LearningData = pd.concat([LearningData, LearningData], axis=0)

TrainData = LearningData.loc[:, 0:len(TrainData.columns) - 1]
LabelData = LearningData.loc[:, LabelData.columns[0]:LabelData.columns[len(LabelData.columns) - 1]]

print('전처리 종료')

###############################################################################################
#데이터 학습
print('학습 시작')

#플랫폼별 학습
for Platform in LabelData.columns:
    startTime = time.time()
    model, result = Model.Learning_KNN(TrainData, LabelData.loc[:, Platform])

    Model.SaveModel(
        ModelType='KNN',
        Model=model,
        result=result,
        PlatForm=str(Platform)
    )

    print(Platform, ": 학습 종료 / ", time.time() - startTime, "sec 소요")

print('학습 종료')