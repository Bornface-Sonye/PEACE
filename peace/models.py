from django.db import models
from django.contrib.auth.hashers import make_password, check_password

class Enforcer(models.Model):
    enforcer_email = models.EmailField(unique=True)
    badgeNo = models.CharField(max_length=100)
    password_hash = models.CharField(max_length=128)  # Store hashed passwords

    def set_password(self, raw_password):
        self.password_hash = make_password(raw_password)

    def check_password(self, raw_password):
        return check_password(raw_password, self.password_hash)
    
    def __str__(self):
        return str(self.enforcer_email)
    
    
class Suspect(models.Model):
    suspect_email = models.EmailField(primary_key=True, help_text="Enter a valid Email Address")
    first_name = models.CharField(max_length=200, help_text="Enter a valid first name")
    last_name = models.CharField(max_length=200, help_text="Enter a valid last name")
    gender = models.CharField(max_length=100, unique=False, help_text="Enter a valid gender")
    date_of_birth = models.DateField(null=True, help_text="Enter a valid date of birth")
    age = models.CharField(max_length=200, help_text="Enter a valid age")

    def __str__(self):
        return str(self.suspect_email)
    
    
class Feedback(models.Model):
    feedback_id = models.EmailField(primary_key=True, help_text="Enter a valid Email Address")
    serial_number =models.ForeignKey(SuspectResponse, on_delete=models.CASCADE)
    date_recorded = models.DateTimeField(auto_now_add=True, help_text="Date of submission")
    feedback = models.CharField(max_length=10, choices=[('Doubt', 'Doubt'), ('Correct', 'Correct')], help_text="Feedback Type ?")

    def __str__(self):
        return str(self.serial_number)
    
    
class County(models.Model):
    county_id = models.models.AutoField(primary_key=True)
    county_name = models.CharField(max_length=50, unique=True, help_text="Enter a The County")

    def __str__(self):
        return str(self.county_name)
 
 
 class News(models.Model):
    news_id = models.EmailField(primary_key=True, help_text="Enter a valid Email Address")
    news_code = models.CharField(max_length=10, choices=[('SEC45', 'SEC45'), ('SEC67', 'SEC67'), ('SEC79', 'SEC79')], help_text="Security Code ?")
    date_recorded = models.DateTimeField(auto_now_add=True, help_text="Date of submission")
    county =models.ForeignKey(County, on_delete=models.CASCADE)
    newsHeader = models.CharField(max_length=50, unique=True, help_text="Enter a Your News Header")
    newsBody   = models.CharField(max_length=250, unique=True, help_text="Enter a Your News Body")
    
    def __str__(self):
        return str(self.newsHeader)
   
    

    
class Case(models.Model):
    case_description = models.CharField(primary_key=True,max_length=200, help_text="Enter a valid case description")
    
    def __str__(self):
        return str(self.case_description)
    
    
class EnforcerCase(models.Model):
    enforcer_email = models.ForeignKey(Enforcer, on_delete=models.CASCADE)
    case_description = models.ForeignKey(Case, on_delete=models.CASCADE)
    
    def __str__(self):
        return str(self.case_description)
    

class SuspectCase(models.Model):
    suspect_email = models.ForeignKey(Suspect, on_delete=models.CASCADE)
    case_description = models.ForeignKey(Case, on_delete=models.CASCADE)
    
    
    def __str__(self):
        return str(self.suspect_email)
    

class SuspectResponse(models.Model):
    testification_id = models.AutoField(primary_key=True, help_text="Enter a valid testification id")
    serial_number = models.CharField(max_length=8, unique=True, help_text="Auto-generated serial number", blank=True)
    case_description = models.ForeignKey(Case, on_delete=models.CASCADE, help_text="Enter a valid Case description")
    suspect_email = models.ForeignKey(Suspect, on_delete=models.CASCADE, help_text="Enter a valid suspect email")
    date_recorded = models.DateTimeField(auto_now_add=True, help_text="Date of submission")
    suspect_Residensce_county = models.CharField(max_length=20, unique=True, help_text="Enter a valid Suspect Residence County")
    incident_county = models.CharField(max_length=20, unique=True, help_text="Enter a Incident County")
    trace = models.CharField(max_length=3, choices=[('Yes', 'Yes'), ('No', 'No')], help_text="Know complainant?")
    know_complainant = models.CharField(max_length=3, choices=[('Yes', 'Yes'), ('No', 'No')], help_text="Know complainant?")
    involved_with_complainant = models.CharField(max_length=3, choices=[('Yes', 'Yes'), ('No', 'No')], help_text="Involved with complainant?")
    recidivist = models.CharField(max_length=3, choices=[('Yes', 'Yes'), ('No', 'No')], help_text="Involved in similar case?")
    
    
    def __str__(self):
        return f"{self.suspect_email}"
