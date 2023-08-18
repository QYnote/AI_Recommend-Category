#전처리 함수모음
import pandas as pd #DataFrame 용

#키워드 리스트 -> Column들로 올린 DataFrame 틀, 데이터는 없음
#해당 제목에 해당되는 키워드들을 dataframe에 추가하여 return
def createInputDf(frame:pd.DataFrame, keywords:list):
    #frame : 기존 학습된 Category List를 Column으로 보낸 DataFrame 빈값
    #title : 기존에 사용된 제목(str) ex) 'aa bb cc'
    insList = []

    for col in frame.columns: #기존 추출 키워드 리스트 호출
        flag = False
        #기존 추출 키워드랑 일치하는 keyword 있는지 확인
        for keyword in keywords:
            if(col == keyword):
                flag = True
                break
        #있으면 1, 없으면 9값 입력
        if(flag):
            insList.append(1)
        else:
            insList.append(0)

    #데이터 넣기
    # pandas 2.0.0 이상
    frame = pd.concat([frame, pd.Series(insList, index=frame.columns).to_frame().T])

    # #pandas 2.0.0 ver미만
    # frame = frame.append(pd.Series(insList, index=frame.columns), ignore_index=True)  

    return frame

