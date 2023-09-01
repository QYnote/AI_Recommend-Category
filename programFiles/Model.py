
import pandas as pd
import numpy as np
import pickle
import joblib   #모델 저장용
import datetime
import os

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
                if((ExceptionWords is not None) and (word in ExceptionWords)):
                    continue
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


#KNN Model 학습
def Learning_KNN(TrainData:pd.DataFrame, LabelData:pd.DataFrame):
    from sklearn.model_selection import train_test_split    #학습데이터 구분용
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn import metrics #학습결과 수치조회용

    #1. 학습, 테스트데이터 분리
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(
        TrainData, LabelData,
        test_size=0.3,  #학습7 : 테스트3
        random_state=10,
    )
    #2. 모델 생성 및 학습
    model = KNeighborsClassifier(n_neighbors = 5)   #모델 설정
    model.fit(X_Train, Y_Train) #모델 학습

    y_pred = model.predict(X_Test)  #모델에 테스트Input 입력
    result = metrics.accuracy_score(Y_Test, y_pred=y_pred)  #정확도 측정
    return model, result

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

def OutputKNN(InputData, minRate:float):
    ResultData = pd.DataFrame(InputData)

    path = os.path.dirname(os.path.realpath(__file__)) + "\\models\\KNN"
    files = os.listdir(path)

    InputData = PreProcess.Vectorise_Apply(PreProcess, InputData)

    for Platform in files:
        #플랫폼별 모델 결과값 추출
        PlfModel = LoadModel(ModelType='KNN', PlatForm=Platform)
        ResultData[Platform] = PlfModel.predict(InputData)

        #해당 데이터 정확도가 일정 수치 이하면 빈값
        ResultData['Rate'] = np.max(PlfModel.predict_proba(InputData), axis=1)
        ResultData.loc[ResultData['Rate'] < minRate, Platform] = np.nan

    #정확도 계산 Column 삭제
    ResultData.drop(['Rate'], axis='columns', inplace=True)

    return ResultData
