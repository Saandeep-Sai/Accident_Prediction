from django.db.models import Count
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404
import datetime
import openpyxl

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.ensemble import VotingClassifier
import warnings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
# Create your views here.
from Remote_User.models import ClientRegister_Model,road_accident_prediction,detection_ratio,detection_accuracy

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')

def Register1(request):

    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city)

        return render(request, 'RUser/Register1.html')
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Predict_Road_Accident_Status(request):
        expense = 0
        kg_price=0
        if request.method == "POST":

            Reference_Number= request.POST.get('Reference_Number')
            State= request.POST.get('State')
            Area_Name= request.POST.get('Area_Name')
            Traffic_Rules_Viaolation= request.POST.get('Traffic_Rules_Viaolation')
            Vechile_Load= request.POST.get('Vechile_Load')
            Time= request.POST.get('Time')
            Road_Class= request.POST.get('Road_Class')
            Road_Surface= request.POST.get('Road_Surface')
            Lighting_Conditions= request.POST.get('Lighting_Conditions')
            Weather_Conditions= request.POST.get('Weather_Conditions')
            Person_Type= request.POST.get('Person_Type')
            Sex= request.POST.get('Sex')
            Age= request.POST.get('Age')
            Type_of_Vehicle= request.POST.get('Type_of_Vehicle')


            df = pd.read_csv('Road_Accidents.csv')
            df
            df.columns
            df.rename(columns={'Label': 'label', 'Reference_Number': 'RId'}, inplace=True)

            def apply_results(label):
                if (label == 0):
                    return "No Accident"
                elif (label == 1):
                    return "Accident"

            df['results'] = df['label'].apply(apply_results)
            results = df['results'].value_counts()
            # df.drop(['Road Surface','Lighting Conditions','Sex','Age','label','Type of Vehicle','Person Type'],axis=1,inplace=True)

            cv = CountVectorizer(lowercase=False)

            y = df['results']
            # X = df.drop("results", axis=1)
            X = df["RId"].apply(str)

            print("X Values")
            print(X)
            print("Labels")
            print(y)

            X = cv.fit_transform(X)

            models = []
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
            X_train.shape, X_test.shape, y_train.shape

            # SVM Model
            print("SVM")
            from sklearn import svm
            lin_clf = svm.LinearSVC()
            lin_clf.fit(X_train, y_train)
            predict_svm = lin_clf.predict(X_test)
            svm_acc = accuracy_score(y_test, predict_svm) * 100
            print(svm_acc)
            print("CLASSIFICATION REPORT")
            print(classification_report(y_test, predict_svm))
            print("CONFUSION MATRIX")
            print(confusion_matrix(y_test, predict_svm))
            models.append(('svm', lin_clf))

            print("KNeighborsClassifier")
            from sklearn.neighbors import KNeighborsClassifier
            kn = KNeighborsClassifier()
            kn.fit(X_train, y_train)
            knpredict = kn.predict(X_test)
            print("ACCURACY")
            print(accuracy_score(y_test, knpredict) * 100)
            print("CLASSIFICATION REPORT")
            print(classification_report(y_test, knpredict))
            print("CONFUSION MATRIX")
            print(confusion_matrix(y_test, knpredict))
            models.append(('KNeighborsClassifier', kn))

            classifier = VotingClassifier(models)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)


            rno = [Reference_Number]
            vector1 = cv.transform(rno).toarray()
            predict_text = lin_clf.predict(vector1)

            pred = str(predict_text).replace("[", "")
            pred1 = pred.replace("]", "")
            prediction = re.sub("[^a-zA-Z]", " ", str(pred1))

            road_accident_prediction.objects.create(Reference_Number=Reference_Number,
            State=State,
            Area_Name =Area_Name,
            Traffic_Rules_Viaolation=Traffic_Rules_Viaolation,
            Vechile_Load=Vechile_Load,
            Time=Time,
            Road_Class=Road_Class,
            Road_Surface=Road_Surface,
            Lighting_Conditions=Lighting_Conditions,
            Weather_Conditions=Weather_Conditions,
            Person_Type=Person_Type,
            Sex=Sex,
            Age=Age,
            Type_of_Vehicle=Type_of_Vehicle,
            SVM=prediction)

            return render(request, 'RUser/Predict_Road_Accident_Status.html',{'objs':prediction})
        return render(request, 'RUser/Predict_Road_Accident_Status.html')

