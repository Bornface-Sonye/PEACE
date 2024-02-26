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
