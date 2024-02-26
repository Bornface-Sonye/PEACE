from django.contrib import admin

from . models import (Suspect, Case, SuspectTestification)

models_to_register = [Suspect, Case, SuspectTestification]

for model in models_to_register:
    admin.site.register(model)
