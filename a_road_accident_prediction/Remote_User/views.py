from django.db.models import Count
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404
import datetime
import openpyxl
import sys
import os
import subprocess
import json

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


def predict_accident_video(request):
    """Video-based accident prediction using CNN classifier."""
    from django.core.files.storage import FileSystemStorage
    from django.conf import settings
    import time
    
    if request.method == 'POST' and request.FILES.get('video'):
        try:
            # Save uploaded video
            video = request.FILES['video']
            fs = FileSystemStorage(location=os.path.join(settings.BASE_DIR, 'uploads', 'videos'))
            
            # Create uploads directory if not exists
            os.makedirs(fs.location, exist_ok=True)
            
            # Generate unique filename
            timestamp = str(int(time.time()))
            filename = f"{timestamp}_{video.name}"
            saved_filename = fs.save(filename, video)
            video_path = fs.path(saved_filename)
            
            # Paths for model and results (go up one level from Django project to workspace root)
            workspace_root = os.path.dirname(settings.BASE_DIR)
            model_path = os.path.join(workspace_root, 'models', 'accident_classifier_mobilenet.h5')
            result_json = os.path.join(settings.BASE_DIR, 'uploads', 'results', f'{timestamp}_prediction.json')
            output_video = os.path.join(settings.BASE_DIR, 'uploads', 'results', f'{timestamp}_annotated.mp4')
            
            # Create results directory
            os.makedirs(os.path.dirname(result_json), exist_ok=True)
            
            # Get Python executable from virtual environment
            if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
                # We're in a virtual environment
                python_exe = sys.executable
            else:
                python_exe = 'python'
            
            # Run prediction script with object detection (in workspace root, not Django project)
            script_path = os.path.join(workspace_root, 'scripts', 'predict_video_with_detection.py')
            
            cmd = [
                python_exe,
                script_path,
                '--model', model_path,
                '--video', video_path,
                '--output', output_video,
                '--json', result_json,
                '--yolo', 'yolov8n.pt',  # Lightweight YOLO model
                '--sample-rate', '3',  # Process every 3rd frame for speed
                '--threshold', '0.5'
            ]
            
            # Run prediction
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                return render(request, 'RUser/predict_video_accident.html', {
                    'error': f'Prediction failed: {result.stderr}'
                })
            
            # Wait a moment for file system to sync
            time.sleep(0.5)
            
            # Debug: Print file paths and existence
            print(f"DEBUG: Checking output video at: {output_video}")
            print(f"DEBUG: Output video exists: {os.path.exists(output_video)}")
            if os.path.exists(output_video):
                print(f"DEBUG: Output video size: {os.path.getsize(output_video)} bytes")
            
            # Load results
            if os.path.exists(result_json):
                with open(result_json, 'r') as f:
                    prediction_data = json.load(f)
                
                # Save prediction to database
                userid = request.session.get('userid')
                if userid:
                    user = ClientRegister_Model.objects.get(id=userid)
                    
                    # Create a record in road_accident_prediction model with video analysis data
                    road_accident_prediction.objects.create(
                        Reference_Number=f"VIDEO_{timestamp}",
                        State='Video Analysis',
                        Area_Name=f"Video: {os.path.basename(video_path)}",
                        Traffic_Rules_Viaolation='N/A',
                        Vechile_Load='N/A',
                        Time=str(prediction_data['video_info'].get('duration_seconds', 0)),
                        Road_Class='Video',
                        Road_Surface='N/A',
                        Lighting_Conditions='Various',
                        Weather_Conditions='N/A',
                        Person_Type='Video Upload',
                        Sex='N/A',
                        Age=f"{prediction_data['statistics']['total_frames']} frames",
                        Type_of_Vehicle=os.path.basename(video_path),
                        SVM=f"{prediction_data['prediction']['prediction']} ({prediction_data['prediction']['confidence']:.1f}%)"
                    )
                
                # Prepare context
                context = {
                    'success': True,
                    'prediction': prediction_data['prediction'],
                    'stats': prediction_data['statistics'],
                    'video_info': prediction_data['video_info'],
                    'video_filename': os.path.basename(video_path),
                    'result_json': os.path.basename(result_json),
                    'has_output_video': os.path.exists(output_video),
                    'result_video': os.path.basename(output_video) if os.path.exists(output_video) else None,
                    'processing_info': prediction_data['processing']
                }
                
                return render(request, 'RUser/predict_video_accident.html', context)
            else:
                return render(request, 'RUser/predict_video_accident.html', {
                    'error': 'Prediction completed but results file not found.'
                })
                
        except subprocess.TimeoutExpired:
            return render(request, 'RUser/predict_video_accident.html', {
                'error': 'Prediction timeout. Video may be too long. Please upload a shorter video (< 30 seconds).'
            })
        except Exception as e:
            import traceback
            return render(request, 'RUser/predict_video_accident.html', {
                'error': f'Error during prediction: {str(e)}',
                'traceback': traceback.format_exc()
            })
    
    return render(request, 'RUser/predict_video_accident.html')


def video_prediction_history(request):
    """View prediction history for the logged-in user."""
    if 'userid' not in request.session:
        return redirect('login')
    
    userid = request.session['userid']
    user = ClientRegister_Model.objects.get(id=userid)
    
    # Get all video predictions (where Type_of_Vehicle contains video file extensions)
    predictions = road_accident_prediction.objects.filter(
        name=user,
        Type_of_Vehicle__icontains='.mp4'
    ).order_by('-id')[:20]  # Last 20 predictions
    
    return render(request, 'RUser/video_prediction_history.html', {
        'predictions': predictions
    })


def serve_video_result(request, result_video):
    """Serve annotated video results with range request support for video streaming."""
    from django.http import FileResponse, Http404, StreamingHttpResponse
    from django.conf import settings
    import mimetypes
    
    # Security: Only allow serving files from results directory
    video_path = os.path.join(settings.BASE_DIR, 'uploads', 'results', result_video)
    
    if not os.path.exists(video_path):
        raise Http404("Video not found")
    
    # Check if user is logged in
    if 'userid' not in request.session:
        raise Http404("Unauthorized")
    
    # Get file size
    file_size = os.path.getsize(video_path)
    content_type = mimetypes.guess_type(video_path)[0] or 'video/mp4'
    
    # Check if this is a range request
    range_header = request.META.get('HTTP_RANGE', '').strip()
    range_match = None
    if range_header:
        import re
        range_match = re.search(r'bytes=(\d+)-(\d*)', range_header)
    
    if range_match:
        # Handle range request for video seeking
        first_byte = int(range_match.group(1))
        last_byte = int(range_match.group(2)) if range_match.group(2) else file_size - 1
        length = last_byte - first_byte + 1
        
        # Open file and seek to position
        file_handle = open(video_path, 'rb')
        file_handle.seek(first_byte)
        
        # Create streaming response
        response = StreamingHttpResponse(
            iter(lambda: file_handle.read(8192), b''),
            status=206,  # Partial Content
            content_type=content_type
        )
        response['Content-Length'] = str(length)
        response['Content-Range'] = f'bytes {first_byte}-{last_byte}/{file_size}'
        response['Accept-Ranges'] = 'bytes'
        response['Content-Disposition'] = f'inline; filename="{result_video}"'
        
        # Add callback to close file when response is consumed
        def file_iterator_wrapper():
            try:
                for chunk in iter(lambda: file_handle.read(8192), b''):
                    yield chunk
            finally:
                file_handle.close()
        
        response.streaming_content = file_iterator_wrapper()
        return response
    else:
        # Normal request - serve entire file
        response = FileResponse(open(video_path, 'rb'), content_type=content_type)
        response['Content-Length'] = str(file_size)
        response['Accept-Ranges'] = 'bytes'
        response['Content-Disposition'] = f'inline; filename="{result_video}"'
        return response

