from django.db import models
from django.contrib.auth.models import User

class Suspect(models.Model):
    suspect_email = models.EmailField(primary_key=True)
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    age = models.PositiveIntegerField()
    drug_test = models.BooleanField()

from django.db import models

class Enforcer(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    enforcer_email = models.EmailField(primary_key=True)
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    password = models.CharField(max_length=100)  # Store hashed password

class Case(models.Model):
    description = models.TextField(primary_key=True)
    suspect_email = models.ForeignKey(Suspect, on_delete=models.CASCADE)
    enforcer_email = models.ForeignKey(Enforcer, on_delete=models.CASCADE)

class FirstQuestioning(models.Model):
    suspect_email = models.ForeignKey(Suspect, on_delete=models.CASCADE)
    case = models.ForeignKey(Case, on_delete=models.CASCADE)
    question1_answer = models.CharField(max_length=200)
    question2_answer = models.BooleanField()
    question3_answer = models.BooleanField()

class SecondQuestioning(models.Model):
    suspect_email = models.ForeignKey(Suspect, on_delete=models.CASCADE)
    case = models.ForeignKey(Case, on_delete=models.CASCADE)
    question1_answer = models.CharField(max_length=200)
    question2_answer = models.BooleanField()
    question3_answer = models.BooleanField()
