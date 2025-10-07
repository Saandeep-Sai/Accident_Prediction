from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)


class road_accident_prediction(models.Model):

    Reference_Number= models.CharField(max_length=300)
    State= models.CharField(max_length=300)
    Area_Name= models.CharField(max_length=300)
    Traffic_Rules_Viaolation= models.CharField(max_length=300)
    Vechile_Load= models.CharField(max_length=300)
    Time= models.CharField(max_length=300)
    Road_Class= models.CharField(max_length=300)
    Road_Surface= models.CharField(max_length=300)
    Lighting_Conditions= models.CharField(max_length=300)
    Weather_Conditions= models.CharField(max_length=300)
    Person_Type= models.CharField(max_length=300)
    Sex= models.CharField(max_length=300)
    Age= models.CharField(max_length=300)
    Type_of_Vehicle= models.CharField(max_length=300)
    SVM= models.CharField(max_length=300)


class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)


