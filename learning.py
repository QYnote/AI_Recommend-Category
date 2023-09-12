print('Package 호출 중')
import pandas as pd #DataFrame 용
import json
import os
import tkinter as tk
from tkinter import filedialog
import time
from sklearn.model_selection import train_test_split    #학습데이터 구분용

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
print(pre.Vectorizer.get_feature_names_out())

#학습, 테스트데이터 분리
X_Train, X_Test, Y_Train, Y_Test = train_test_split(
    TrainData, LabelData,
    test_size=0.3,  #학습7 : 테스트3
    random_state=10,
)

#결측치 제거
LearningData = pd.concat([X_Train, Y_Train], axis=1).dropna()

#학습데이터 증가
doubleCnt = 0 #데이터 복사 횟수
while(LearningData.shape[0] < json_obj["MinLearningDataCnt"]):
    LearningData = pd.concat([LearningData, LearningData], axis=0)

    doubleCnt += 1

TrainData = LearningData.loc[:, 0:len(TrainData.columns) - 1]
LabelData = LearningData.loc[:, LabelData.columns[0]:LabelData.columns[len(LabelData.columns) - 1]]

print('전처리 종료')

###############################################################################################
#데이터 학습
print('학습 시작')

#플랫폼별 학습
learning = Model.Learning(json_obj["LearningModel"])

for Platform in json_obj["LearningPlatform"]:
    startTime = time.time()

    if(json_obj["LearningModel"] == 'RandomForest'):
        model, result = learning.Learning_RandomForest(TrainData, LabelData.loc[:, Platform], X_Test, Y_Test.loc[:, Platform])
    elif(json_obj["LearningModel"] == 'KNN'):
        model, result = learning.Learning_KNN(TrainData, LabelData.loc[:, Platform], X_Test, Y_Test.loc[:, Platform], 5 * (2 ** doubleCnt))

    Model.SaveModel(
        ModelType=json_obj["LearningModel"],
        PlatForm=str(Platform),
        Model=model,
        result=result
    )

    print(Platform, ": 학습 종료 / ", time.time() - startTime, "sec 소요")

print('학습 종료')