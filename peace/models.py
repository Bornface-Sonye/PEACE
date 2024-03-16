from django.db import models
from django.contrib.auth.hashers import make_password, check_password


class BadgeNumber(models.Model):
    badge_no = models.DecimalField(primary_key=True, max_digits=6, decimal_places=0, help_text="Enter a valid Badge Number")
    first_name = models.CharField(max_length=200, help_text="Enter a valid First Name")
    last_name = models.CharField(max_length=200, help_text="Enter a valid Last Name")
    
    def __str__(self):
        return str(self.badge_no)


class Enforcer(models.Model):
    officer_id = models.AutoField(primary_key=True)
    badge_no =models.ForeignKey(BadgeNumber, on_delete=models.CASCADE)
    password = models.CharField(max_length=128)  # Store hashed passwords

    def set_password(self, raw_password):
        self.password = make_password(raw_password)

    def check_password(self, raw_password):
        return check_password(raw_password, self.password)
    
    def __str__(self):
        return str(self.badge_no)
    
    
class DepartmentNumber(models.Model):
    dep_no = models.CharField(primary_key=True, max_length=10, help_text="Enter a valid Department Number")
    dep_name = models.CharField(max_length=200, help_text="Enter a valid First Name")
    dep_head = models.CharField(max_length=200, help_text="Enter a valid Last Name")
    
    def __str__(self):
        return str(self.dep_no)
    

class Department(models.Model):
    dep_id = models.AutoField(primary_key=True)
    dep_no = models.ForeignKey(DepartmentNumber, on_delete=models.CASCADE)
    password = models.CharField(max_length=128)  # Store hashed passwords

    def set_password(self, raw_password):
        self.password = make_password(raw_password)

    def check_password(self, raw_password):
        return check_password(raw_password, self.password)
    
    def __str__(self):
        return str(self.dep_no)  
    
    
class Suspect(models.Model):
    unique_id = models.CharField(primary_key=True,max_length=10, unique=True, help_text="Enter a valid Suspect Unique Identifier")
    first_name = models.CharField(max_length=200, help_text="Enter a valid First Name")
    last_name = models.CharField(max_length=200, help_text="Enter a valid Last Name")
    gender = models.CharField(max_length=100, unique=False, help_text="Enter a valid Gender")
    date_of_birth = models.DateField(null=True,  help_text="Enter a valid Date of Birth")
    age = models.CharField(max_length=200, help_text="Enter a valid Age")

    def __str__(self):
        return str(self.unique_id)
    
    
class County(models.Model):
    county_id = models.AutoField(primary_key=True)
    county_name = models.CharField(max_length=50, unique=True, help_text="Enter a The County")

    def __str__(self):
        return str(self.county_name)
 
 
class New(models.Model):
    news_id = models.AutoField(primary_key=True)
    news_code = models.CharField(max_length=10, choices=[('SEC45', 'SEC45'), ('SEC67', 'SEC67'), ('SEC79', 'SEC79')], help_text="Security Code ?")
    date_recorded = models.DateTimeField(auto_now_add=True, help_text="Date of submission")
    county = models.ForeignKey(County, on_delete=models.CASCADE, help_text="Enter a Your News County")
    news_header = models.CharField(max_length=50, unique=True, help_text="Enter a Your News Header")
    news_body   = models.CharField(max_length=250, unique=True, help_text="Enter a Your News Body")
    
    def __str__(self):
        return str(self.newsHeader)
   
class CaseCollection(models.Model):
    case_id = models.CharField(primary_key=True, max_length=10, unique=True, help_text="Enter a valid Case Identifier")
    case_description = models.CharField(max_length=200, help_text="Enter a valid Case Description")
    
    def __str__(self):
        return str(self.case_id)  

    
class Case(models.Model):
    case_description = models.OneToOneField(CaseCollection, on_delete=models.CASCADE)
    
    def __str__(self):
        return str(self.case_description)
    
    
    
class EnforcerCase(models.Model):
    badeg_no = models.ForeignKey(Enforcer, on_delete=models.CASCADE)
    case_description = models.ForeignKey(Case, on_delete=models.CASCADE)
    
    def __str__(self):
        return str(self.case_description)
    

class SuspectCase(models.Model):
    unique_id = models.ForeignKey(Suspect, on_delete=models.CASCADE)
    case_description = models.ForeignKey(Case, on_delete=models.CASCADE)
    
    
    def __str__(self):
        return str(self.unique_id)
    

class SuspectResponse(models.Model):
    testification_id = models.AutoField(primary_key=True, help_text="Enter a valid testification id")
    case_description = models.ForeignKey(Case, on_delete=models.CASCADE, help_text="Enter a valid Case description")
    unique_id = models.ForeignKey(Suspect, on_delete=models.CASCADE, help_text="Enter a valid Suspect Identifier")
    serial_number = models.CharField(max_length=8, unique=True, help_text="Auto-generated serial number", blank=True)
    date_recorded = models.DateTimeField(auto_now_add=True, help_text="Date of submission")
    suspect_Residensce_county = models.CharField(max_length=20, unique=True, help_text="Enter a valid Suspect Residence County")
    incident_county = models.CharField(max_length=20, unique=True, help_text="Enter the Incident County")
    trace = models.CharField(max_length=3, choices=[('Yes', 'Yes'), ('No', 'No')], help_text="Any strong Trace of Suspect in Crime Scene ?")
    know_complainant = models.CharField(max_length=3, choices=[('Yes', 'Yes'), ('No', 'No')], help_text="Know complainant?")
    involved_with_complainant = models.CharField(max_length=3, choices=[('Yes', 'Yes'), ('No', 'No')], help_text="Involved with complainant?")
    recidivist = models.CharField(max_length=3, choices=[('Yes', 'Yes'), ('No', 'No')], help_text="Involved in similar case?")
    
    
    def __str__(self):
        return f"{self.unique_id}"
    
    
class Feedback(models.Model):
    feedback_id = models.AutoField(primary_key=True)
    serial_number =models.ForeignKey(SuspectResponse, on_delete=models.CASCADE)
    date_recorded = models.DateTimeField(auto_now_add=True, help_text="Date of submission")
    feedback = models.CharField(max_length=10, choices=[('Doubt', 'Doubt'), ('Correct', 'Correct')], help_text="Feedback Type ?")

    def __str__(self):
        return str(self.serial_number)
