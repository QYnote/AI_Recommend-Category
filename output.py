#데이터 입력해서 결과값 받기
print('Package 불러오기 시작')
import pandas as pd #DataFrame 용
import os
import json

from programFiles import Model

import tkinter as tk
from tkinter import filedialog

print('Package 불러오기 종료')


#json파일 불러오기 : 이 파일과 같은 폴더에 있을 것
print('json 정보 가져오기')

path = os.path.dirname(os.path.realpath(__file__)) + "\\" + 'ConstData.json'

with open(path, encoding='utf-8') as f:
    json_obj = json.load(f)

json_obj = json_obj["Run"]

print('json 정보 가져오기 종료')

#엑셀파일 진행
#적용할 파일 불러오기
print('파일 불러오기')
root = tk.Tk()
root.withdraw() #Python 기본 윈도우 숨기기
openFilePath = filedialog.askopenfilename(
    filetypes=(("xlsx File", "*.xlsx"), ("xls File", "*.xls")),
    title='적용파일 불러오기'
)
# openFilePath = "D:/00.Storage/Study/Programing/Project/AI_Recommend-Category/10000allchem_ESELLERS_.xlsx"
if(openFilePath == ''):
    exit()

dfExcel = pd.read_excel(openFilePath)
df = pd.DataFrame(dfExcel, columns= [json_obj["runUseColumn"]]) #사용하는 Column만 추출

print('파일 불러오기 종료')

#결과물 출력
print("결과물 출력 시작")

OutputData = Model.OutputKNN(df, json_obj["OutputPlatform"], json_obj["MinAccuracy"])

print("결과물 출력 종료")

#결과물 저장
print("결과물 저장 시작")
SaveFilePath = filedialog.asksaveasfilename(
    filetypes=(("xlsx File", "*.xlsx"), ("xls File", "*.xls")),
    title='결과파일 저장',
    defaultextension = '.xlsx'
)
if(SaveFilePath == None):
    exit()

OutputData.to_excel(SaveFilePath, index=False)

print("결과물 저장 종료")