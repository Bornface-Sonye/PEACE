<<<<<<< HEAD
from django.db import models
    
class Suspect(models.Model):
    email_address = models.EmailField(primary_key=True, help_text="Enter a valid Email Address")
    first_name = models.CharField(max_length=200, help_text="Enter a valid first name")
    last_name = models.CharField(max_length=200, help_text="Enter a valid last name")
    gender = models.CharField(max_length=100, unique=False, help_text="Enter a valid gender")
    date_of_birth = models.DateField(null=True, help_text="Enter a valid date of birth")
    drug_test = models.BooleanField(default=False, help_text="Enter a valid drug test")
    age = models.CharField(max_length=200, help_text="Enter a valid age")

    def __str__(self):
        return str(self.email_address)
    
class Case(models.Model):
    case_description = models.CharField(primary_key=True,max_length=200, help_text="Enter a valid case description")
    
    def __str__(self):
        return str(self.case_description)

class SuspectTestification(models.Model):
    testification_id = models.AutoField(primary_key=True, help_text="Enter a valid testification id")
    email_address= models.ForeignKey(Suspect, on_delete=models.CASCADE, help_text="Enter a valid suspect id")
    case_description = models.ForeignKey(Case, on_delete=models.CASCADE, help_text="Enter a valid Case description")
    answer_1 = models.CharField(max_length=100, help_text="Enter a valid crime location")
    answer_2 = models.CharField(max_length=100, help_text="Enter a valid crime victim")
    
    def __str__(self):
        return f"{self.email_address}"
=======
from django.db import models
    
class User(models.Model):
    email_address = models.EmailField(primary_key=True, help_text="Enter a valid Email Address")
    user_password = models.CharField(max_length=200, help_text="Enter a valid password")
    first_name = models.CharField(max_length=200, help_text="Enter a valid first name")
    last_name = models.CharField(max_length=200, help_text="Enter a valid last name")
    gender = models.CharField(max_length=100, unique=False, help_text="Enter a valid gender")
    date_of_birth = models.DateField(null=True, help_text="Enter a valid date of birth")
    drug_test = models.BooleanField(default=False, help_text="Enter a valid drug test")
    age = models.CharField(max_length=200, help_text="Enter a valid age")
    
    
    def __str__(self):
        return str(self.email_address)
    
class Case(models.Model):
    case_description = models.CharField(primary_key=True,max_length=200, help_text="Enter a valid case description")
    case_status = models.CharField(max_length=100, unique=False, help_text="Enter a valid case status")
    
    def __str__(self):
        return str(self.case_description)

class Enforcer(models.Model):
    enforcer_id = models.AutoField(primary_key=True)
    email_address =  models.ForeignKey(User, on_delete=models.CASCADE, help_text="Enter a valid Email Address")
    case_description = models.ForeignKey(Case, on_delete=models.CASCADE, help_text="Enter a valid Case Description")
    
    def __str__(self):
        return str(self.enforcer_id)

class Witness(models.Model):
    witness_id = models.AutoField(primary_key=True)
    email_address = models.ForeignKey(User, on_delete=models.CASCADE, help_text="Enter a valid email address")
    case_description =  models.ForeignKey(Case, on_delete=models.CASCADE, help_text="Enter a valid case description")
    testification_status = models.BooleanField(default=False, help_text="Enter a valid testification status")
    
    def __str__(self):
        return str(self.witness_id)

class Suspect(models.Model):
    suspect_id = models.AutoField(primary_key=True)
    email_address = models.ForeignKey(User, on_delete=models.CASCADE, help_text="Enter a valid email address")
    case_description = models.ForeignKey(Case, on_delete=models.CASCADE, help_text="Enter a valid case description")
    testification_status = models.BooleanField(default=False, help_text="Enter a valid testification status")
    
    def __str__(self):
        return str(self.suspect_id)

class Statement(models.Model):
    statement_id = models.AutoField(primary_key=True)
    enforcer_id = models.ForeignKey(Enforcer, on_delete=models.CASCADE, help_text="Enter a valid Enforcer ID")
    case_description = models.ForeignKey(Case, on_delete=models.CASCADE, help_text="Enter a valid Case description")
    crime_location = models.CharField(max_length=200, help_text="Enter a valid Crime location")
    crime_victim = models.CharField(max_length=200, help_text="Enter a valid crime victim")
    crime_incident_date = models.DateField(help_text="Enter a valid crime incident date")
    crime_incident_time = models.TimeField(help_text="Enter a valid crime incident time")
    
    def __str__(self):
        return f"ID: {self.statement_id}, Enforcer ID: {self.enforcer_id}, Case Description: {self.case_description}, Crime Location: {self.crime_location}, Crime Victim: {self.crime_victim}, Incident Date: {self.crime_incident_date}, Incident Time: {self.crime_incident_time}"

class WitnessTestification(models.Model):
    testification_id = models.AutoField(primary_key=True)
    witness_id = models.ForeignKey(Witness, on_delete=models.CASCADE, help_text="Enter a valid witness id")
    case_description = models.ForeignKey(Case, on_delete=models.CASCADE, help_text="Enter a valid Case description")
    case_location = models.CharField(max_length=200, help_text="Enter a valid case location")
    case_victim = models.CharField(max_length=200, help_text="Enter a valid crime victim")
    case_incident_date = models.DateField(help_text="Enter a valid case incident date")
    case_incident_time = models.TimeField(help_text="Enter a valid case incident time")
    case_potential_suspect = models.CharField(max_length=200, help_text="Enter a valid case potential suspect")
    
    def __str__(self):
        return f"ID: {self.testification_id}, Witness ID: {self.witness_id}, Case Description: {self.case_description}, Case Location: {self.case_location}, Case Victim: {self.case_victim}, Incident Date: {self.case_incident_date}, Incident Time: {self.case_incident_time}, Potential Suspect: {self.case_potential_suspect}"

class SuspectTestification(models.Model):
    testification_id = models.AutoField(primary_key=True, help_text="Enter a valid testification id")
    suspect_id = models.ForeignKey(Suspect, on_delete=models.CASCADE, help_text="Enter a valid suspect id")
    case_description = models.ForeignKey(Case, on_delete=models.CASCADE, help_text="Enter a valid Case description")
    crime_location = models.CharField(max_length=100, help_text="Enter a valid crime location")
    crime_victim = models.CharField(max_length=100, help_text="Enter a valid crime victim")
    crime_incident_date = models.DateField(help_text="Enter a valid crime incident date")
    crime_incident_time = models.TimeField(help_text="Enter a valid crime incident time")
    
    def __str__(self):
        return f"ID: {self.testification_id}, Suspect ID: {self.suspect_id}, Case Description: {self.case_description}, Crime Location: {self.crime_location}, Crime Victim: {self.crime_victim}, Incident Date: {self.crime_incident_date}, Incident Time: {self.crime_incident_time}"
    
    
class SentimentAnalyser(models.Model):
    analyser_id = models.AutoField(primary_key=True)
    suspect_id = models.ForeignKey(Suspect, on_delete=models.CASCADE,help_text="Enter a valid suspect id")
    emotion_score = models.FloatField(help_text="Enter a valid emotion score")
    confidence_score = models.FloatField(help_text="Enter a valid confidence score")
    consistency_score = models.FloatField(help_text="Enter a valid consitency score")
    
    def __str__(self):
        return f"ID: {self.analyser_id}, Suspect ID: {self.suspect_id}, Emotion Score: {self.emotion_score}, Confidence Score: {self.confidence_score}, Consistency Score: {self.consistency_score}"

class Prediction(models.Model):
    prediction_id = models.AutoField(primary_key=True)
    suspect_id = models.ForeignKey(Suspect, on_delete=models.CASCADE, help_text="Enter a valid suspect id")
    prediction_result  = models.CharField(max_length=200, unique=False, help_text="Enter a valid result")
    
    def __str__(self):
        return f"ID: {self.prediction_id}, Suspect ID: {self.suspect_id}, Prediction Result: {self.prediction_result}"
>>>>>>> 7ff2cbcfdbc7181a4fe291243adbdb7c4db75281
