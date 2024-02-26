<<<<<<< HEAD
from django.contrib import admin

from . models import (Suspect, Case, SuspectTestification)

models_to_register = [Suspect, Case, SuspectTestification]

for model in models_to_register:
    admin.site.register(model)
=======
from django.contrib import admin

from . models import (User, Case, Enforcer, Witness, Suspect, Statement, WitnessTestification, SuspectTestification, SentimentAnalyser, Prediction)

models_to_register = [User, Case, Enforcer, Witness, Suspect, Statement, WitnessTestification, SuspectTestification, SentimentAnalyser, Prediction]

for model in models_to_register:
    admin.site.register(model) 
>>>>>>> 7ff2cbcfdbc7181a4fe291243adbdb7c4db75281
