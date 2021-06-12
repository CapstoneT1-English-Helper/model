import flask
import os
import cv2
import pandas as pd
import pickle
from flask import Flask, render_template, request
import sys
import numpy as np
import torch
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

application = Flask(__name__)


@application.route("/")
@application.route("/index")
def index():
    return flask.render_template('index.html')
        

#데이터 예측 처리
@application.route('/predict', methods=['POST'])
def make_prediction():
    if request.method=='POST':
        #업로드 파일 처리 분기
        file=request.files['image']
        #if not file: return render_template('index.html', label="No Files")             
        # 그걸 그대로 적당한 파일명 줘서 저장
        file.save('/workspace/project/yolov5/data/images/imagedata.jpg')
        #if not file: return render_template('index.html', label="No Files")
        
        # Model
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        
        data_frame = pd.read_csv('/workspace/project/data_frame.csv')

        # 추출한 객체의 정확도, 위치를 함께 리턴하는 함수
        objDetec_name = pd.DataFrame(columns = ['name','pred','x1','x2','y1','y2'])
        def printingPredResult(self):
                n = 0
                for i, (im, pred) in enumerate(zip(self.imgs, self.pred)) :
                    if pred is not None :
                        for *box, conf, cls in pred:  # xyxy, confidence, class
                            configuration_size = f'{conf:.2f}'
                            objDetec_name.loc[n] = [self.names[int(cls)], configuration_size, box[0].numpy(), box[2].numpy(), box[1].numpy(), box[3].numpy()]
                            n += 1

                return objDetec_name

        # image 파일 받아오기
        img_name = '/workspace/project/yolov5/data/images/imagedata.jpg'

        img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        results = model(img)
        results.save()
        
        
        

        ___ = printingPredResult(results)  # object, pred 값으로 이루어진 dataframe 받아오기

        # 정확도가 더 높은 순서대로 저장되므로 가장 먼저 나오는 객체 중 person이 아닌 값을 O(목적어)로 주기
        for j in range(len(___)) :
            if ___.loc[j,'name'] != 'person' :
                data_frame.loc[0,'O_x1'] = ___.loc[j,'x1']
                data_frame.loc[0,'O_x2'] = ___.loc[j,'x2']
                data_frame.loc[0,'O_y1'] = ___.loc[j,'y1']
                data_frame.loc[0,'O_y2'] = ___.loc[j,'y2']
                O_object = ___.loc[j,'name']
                O_ = 'O_' + ___.loc[j,'name']
                for i in data_frame.columns :
                    if O_ == i :
                        data_frame[i] = 1
                break
            else : 
                continue

        for j in range(len(___)) :
            if ___.loc[j,'name'] == 'person' :
                data_frame.loc[0,'S_x1'] = ___.loc[j,'x1']
                data_frame.loc[0,'S_x2'] = ___.loc[j,'x2']
                data_frame.loc[0,'S_y1'] = ___.loc[j,'y1']
                data_frame.loc[0,'S_y2'] = ___.loc[j,'y2']
                S_object = 'person'
                data_frame['S_person'] = 1
                break
            else : 
                continue

        data_frame.to_csv("/workspace/project/yolov5/test_data.csv")  # csv로 저장      
        
        model_from_joblib = joblib.load('/workspace/project/model/english_helper.pkl')
        
        #입력 받은 이미지 예측
        prediction=model_from_joblib.predict(data_frame)

        #예측 값을 1차원 배열로부터 확인 가능한 문자열로 변환
        label = 'The ' + S_object + ' ' +str(np.squeeze(prediction)) + ' ' + O_object + '.'
                
        #결과 리턴
        return flask.render_template('index.html', label=label)

if __name__ == "__main__":
    #모델 로드
    #ml/model.py 선 실행 후 생성
    #model_from_joblib = joblib.load('/workspace/project/model/english_helper.pkl')   
    #Flask 서비스 스타트
    application.debug=True
    application.run(host='0.0.0.0', port=5000)
