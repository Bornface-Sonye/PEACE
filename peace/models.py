from django.db import models
from django.contrib.auth.hashers import make_password, check_password


class BadgeNumber(models.Model):
    badge_no = models.DecimalField(primary_key=True, max_digits=6, decimal_places=0, help_text="Enter a valid Badge Number")
    first_name = models.CharField(max_length=200, help_text="Enter a valid First Name")
    last_name = models.CharField(max_length=200, help_text="Enter a valid Last Name")
    email_address = models.EmailField(max_length=200,default=" ", help_text="Enter a valid Email Address")
    
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
    suspect_name = models.CharField(max_length=200, default=" ", help_text="Enter Full Suspect Name")
    gender = models.CharField(max_length=100, unique=False, help_text="Enter a valid Gender")
    date_of_birth = models.DateField(null=True,  help_text="Enter a valid Date of Birth")
    age = models.CharField(max_length=200, help_text="Enter a valid Age")

    def __str__(self):
        return str(self.unique_id)
    
    
class County(models.Model):
    county_name = models.CharField(primary_key =True, max_length=50, unique=True, help_text="Enter The County")

    def __str__(self):
        return str(self.county_name)
 
   
class CaseCollection(models.Model):
    case_id = models.CharField(primary_key=True, max_length=10, unique=True, help_text="Enter a valid Case Identifier")
    case_description = models.CharField(max_length=200, help_text="Enter a valid Case Description")
    case_victim = models.CharField(max_length=200,default=" ", help_text="Enter a valid Case Victim")
    
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
    date_recorded = models.DateTimeField(auto_now_add=True, help_text="Date of submission", blank=True)
    trace = models.CharField(max_length=3, choices=[('Yes', 'Yes'), ('No', 'No')], help_text="Any strong Trace of Suspect in Crime Scene?")
    know_complainant = models.CharField(max_length=3, choices=[('Yes', 'Yes'), ('No', 'No')], help_text="Know complainant?")
    involved_with_complainant = models.CharField(max_length=3, choices=[('Yes', 'Yes'), ('No', 'No')], help_text="Involved with complainant?")
    recidivist = models.CharField(max_length=3, choices=[('Yes', 'Yes'), ('No', 'No')], help_text="Involved in similar case?")
    question1 = models.CharField(max_length=250, default="", help_text="Can you describe what happened on the incident day providing as many details as possible")
    question2 = models.CharField(max_length=250, default="", help_text="Can you think of any reason why someone would lie about this incident ? If yes, explain.")
    question3 = models.CharField(max_length=250, default="", help_text="Are you aware of any other complaints made by the accuser ? If yes, State.")
    query1 = models.CharField(max_length=250, default="", help_text="Provide detailed description of the incident day's events.")
    query2 = models.CharField(max_length=250, default="", help_text="Any motive for dishonesty about the incident? Please elaborate if applicable.")
    query3 = models.CharField(max_length=250, default="", help_text="Have you heard of any additional complaints filed by the accuser? If so, specify.")
    
    
    
    def __str__(self):
        return f"{self.unique_id}"
    
    
# models.py

class PasswordResetToken(models.Model):
    username = models.ForeignKey(BadgeNumber,default=" ", on_delete=models.CASCADE)
    token = models.CharField(max_length=32, default=" ",)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Token for {self.username}"


