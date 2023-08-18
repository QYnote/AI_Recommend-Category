from sklearn.model_selection import train_test_split    #학습데이터 구분용
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics #학습결과 수치조회용
import os

#최근 모델 가져오기
def getRecentModel():
    path = os.path.dirname(os.path.realpath(__file__)) + "\\models"
    files = os.listdir(path)

    fileName = files[len(files) - 1]

    return path + "\\" + fileName

#KNN Model 학습
def learning_KNN(TrainData, LabelData):
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(
        TrainData, LabelData,
        test_size=0.3,  #학습7 : 테스트3
        random_state=10,
    )

    model = KNeighborsClassifier(n_neighbors = 5)   #모델 설정
    model.fit(X_Train, Y_Train) #모델 학습

    y_pred = model.predict(X_Test)  #모델에 Input 입력
    result = metrics.accuracy_score(Y_Test, y_pred=y_pred)  #정확도 측정

    return model, result
