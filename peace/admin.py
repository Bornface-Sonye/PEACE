from django.contrib import admin

from . models import (Suspect, Case, SuspectResponse, Enforcer, EnforcerCase, SuspectCase)

models_to_register = [Suspect, Case, SuspectResponse, Enforcer, EnforcerCase, SuspectCase]

for model in models_to_register:
    admin.site.register(model)
