from django.contrib import admin
from .models import Suspect, Case, SuspectResponse, Department, BadgeNumber, Feedback
from .models import Enforcer, CaseCollection, EnforcerCase, SuspectCase, New, County, DepartmentNumber

models_to_register = [Suspect, Case, SuspectResponse, Department, BadgeNumber, Feedback, Enforcer, CaseCollection, EnforcerCase, SuspectCase, New, County, DepartmentNumber]

for model in models_to_register:
    admin.site.register(model)
