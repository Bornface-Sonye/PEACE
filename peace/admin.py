from django.contrib import admin
from .models import Suspect, Case, SuspectResponse, Department, BadgeNumber
from .models import Enforcer, CaseCollection, EnforcerCase, SuspectCase, County, DepartmentNumber

models_to_register = [Suspect, Case, SuspectResponse, Department, BadgeNumber, Enforcer, CaseCollection, EnforcerCase, SuspectCase, County, DepartmentNumber]

for model in models_to_register:
    admin.site.register(model)
