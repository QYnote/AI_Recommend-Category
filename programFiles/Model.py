
import pandas as pd
import numpy as np
import pickle
import joblib   #모델 저장용
import datetime
import os

from sklearn.model_selection import train_test_split    #학습데이터 구분용

ModelTypeList = ['KNN', 'RandomForest']

class PreProcess:
    from konlpy.tag import Komoran
    from sklearn.feature_extraction.text import CountVectorizer

    komoran = Komoran()
    Vectorizer = CountVectorizer(
        min_df = 2  #최소 사용 단어량
    )

    #키워드 형태소 구분
    def KeywordCheck(self, Sentencse:str, ExceptionWords:list = None):

        #1. 형태소 구분
        words = PreProcess.komoran.pos(str(Sentencse))

        #2. Title의 키워드들 점검
        keywords = []
        UseTag = ["NNG", "NNP", "NP"]   #입력할 구분

        for wordInfo in words:
            word = str(wordInfo[0])
            wordTag = wordInfo[1]

            if(wordTag in UseTag):    #입력할 Tag에 속하면 입력
                #제외단어 스킵
                if((ExceptionWords is not None) and (word in ExceptionWords)):
                    continue
                #숫자면 스킵
                if(word.isdecimal()): continue

                keywords.append(word)

        #3. Vector화하기위해 1row 1문장처리
        rstSentencse = ''
        for word in keywords:
            rstSentencse = rstSentencse + ' ' + word
        
        return rstSentencse

    #Vectorise 학습 시킬때
    def Vectorise_Learn(self, SentenceList:list):
        vect = PreProcess.Vectorizer

        #1. Vector화
        vectModel = vect.fit_transform(SentenceList)

        #2. Vectorizer 구조(Dictionary) 저장
        joblib.dump(vect.vocabulary_, os.path.dirname(os.path.realpath(__file__)) + "\\Vect_Dict.p")

        return pd.DataFrame(vectModel.toarray())

    #Vectorise 적용시킬때
    def Vectorise_Apply(self, SentenceList):
        vect = PreProcess.Vectorizer
        SentenceList = pd.DataFrame(SentenceList)

        #1. Vector 구조 호출
        vect.vocabulary = joblib.load(os.path.dirname(os.path.realpath(__file__)) + "\\Vect_Dict.p")

        vectData = []
        for Sentence in SentenceList.loc[:, SentenceList.columns[0]]:
            vectData.append(vect.transform([Sentence]).toarray()[0])

        # print(vectData)

        return pd.DataFrame(vectData)

    #플랫폼Data 구조 변경
    def Platform_Modify(self, df:pd.DataFrame):
        LabelList = []

        for row in df:
            PlatformDataList = str(row).splitlines()    #1row에 존재하는 PlatForm들

            PlatformDict = {}
            for PlatformData in PlatformDataList:
                #*가 2개이상들어가서 split 값이 많아지는 경우
                if(PlatformData.count('*') > 1):
                    parts = PlatformData.split('*')
                    key = parts[0]
                    value = parts[1]

                    PlatformDict[key] = value

                #일반적인 경우
                elif(PlatformData.count('*') == 1):
                    key, value = PlatformData.split('*')

                    PlatformDict[key] = value

            LabelList.append(PlatformDict)
        return pd.DataFrame(LabelList)

class Learning:
    from sklearn import metrics #학습결과 수치조회용

    def __init__(self, ModelName:str):
        if(ModelName == 'KNN'):
            from sklearn.neighbors import KNeighborsClassifier  #KNN : Output Multi Column 가능

            self.model = KNeighborsClassifier()   #모델 설정

        elif(ModelName == 'RandomForest'):
            from sklearn.ensemble import RandomForestClassifier #RandomForest : Output Multi Column 불가능

            self.model = RandomForestClassifier(
                n_estimators=20,
                min_samples_leaf = 4,
                max_depth=20
            )

    #KNN Model 학습
    def Learning_KNN(self, TrainData:pd.DataFrame, LabelData:pd.DataFrame, TestInput:pd.DataFrame, TestTrue=pd.DataFrame, n_neighbors = 5):
        #모델 생성 및 학습
        self.model.n_neighbors = n_neighbors   #모델 설정
        self.model.fit(TrainData, LabelData) #모델 학습

        y_pred = self.model.predict(TestInput)  #모델에 테스트Input 입력
        result = self.metrics.accuracy_score(y_true=TestTrue, y_pred=y_pred)  #정확도 측정
        return self.model, result

    #RandomForest Model 학습
    def Learning_RandomForest(self, TrainData:pd.DataFrame, LabelData:pd.DataFrame, TestInput:pd.DataFrame, TestTrue=pd.DataFrame):
        #모델 생성 및 학습
        #모델 설정
        self.model.fit(TrainData, LabelData) #모델 학습

        y_pred = self.model.predict(TestInput)  #모델에 테스트Input 입력
        result = self.metrics.accuracy_score(y_true=TestTrue, y_pred=y_pred)  #정확도 측정
        return self.model, result

#모델 저장
def SaveModel(ModelType:ModelTypeList, Model, result, PlatForm:str):
    try:
        #model 폴더 존재 확인
        path = os.path.dirname(os.path.realpath(__file__)) + "\\models"
        if not os.path.exists(path):    
            os.makedirs(path)

        #model\KNN 폴더 존재 확인
        path = path + "\\" + ModelType
        if not os.path.exists(path):
            os.makedirs(path)
        
        #model\KNN\각 프랫폼 폴더 존재 확인
        path = path + "\\" + PlatForm
        if not os.path.exists(path):
            os.makedirs(path)

    except OSError:
        print("Error: Failed to create the directory.")

    #모델명 : 모델명_플랫폼_날짜_시분_정확도
    modelName = ModelType + "_" + PlatForm + "_" + str(datetime.datetime.now().strftime('%Y%m%d_%H%M')) + "_" + '{:.2f}'.format(result)

    joblib.dump(Model, path + "\\" + modelName + ".pkl")


#모델 불러오기
def LoadModel(ModelType:ModelTypeList, PlatForm:str):
    path = os.path.dirname(os.path.realpath(__file__)) + "\\models\\" + ModelType + "\\" + PlatForm
    files = os.listdir(path)

    fileName = files[len(files) - 1]

    return joblib.load(path + "\\" + fileName)

#OutputData 내보내기
def OutputDataFrame(ModelName:str, InputData, PlatformList:list, minRate:float):
    ResultData = pd.DataFrame(InputData)

    InputData = PreProcess.Vectorise_Apply(PreProcess, InputData)

    for Platform in PlatformList:
        #플랫폼별 모델 결과값 추출
        PlfModel = LoadModel(ModelType=ModelName, PlatForm=Platform)
        ResultData[Platform] = PlfModel.predict(InputData)

        #해당 데이터 정확도가 일정 수치 이하면 빈값
        RstRateList = PlfModel.predict_proba(InputData)
        ResultData['Rate' + Platform] = np.max(RstRateList, axis=1)
        ResultData.loc[ResultData['Rate' + Platform] < minRate, Platform] = np.nan

    # print(ResultData)
    #정확도 계산 Column 삭제
    # ResultData.drop(['Rate'], axis='columns', inplace=True)

    return ResultData
