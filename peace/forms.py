<<<<<<< HEAD
from django import forms
from .models import Suspect, Case, SuspectTestification

class AnswerForm(forms.ModelForm):
    email_address = forms.ModelChoiceField(
        queryset=Suspect.objects.all(),
        required=True,
        label='Email Address',
        widget=forms.Select(attrs={'class': 'blue-input-box'}),
    )
    case_description = forms.ModelChoiceField(
        queryset=Case.objects.all(),
        required=True,
        label='Case Description',
        widget=forms.Select(attrs={'class': 'blue-input-box'}),
    )

    class Meta:
        model = SuspectTestification
        fields = ['email_address', 'case_description', 'answer_1', 'answer_2']
        labels = {
            'email_address': 'Email Address',
            'case_description': 'Case Description',
            'answer_1': 'Question One Answer',
            'answer_2': 'Question Two Answer',
        }
        widgets = {
            'answer_1': forms.TextInput(attrs={'class': 'blue-input-box'}),
            'answer_2': forms.TextInput(attrs={'class': 'blue-input-box'}),
        }

class InterrogatorReportForm(forms.Form):
    email_address = forms.ModelChoiceField(
        queryset=Suspect.objects.all(),
        required=True,
        label='Email Address',
        widget=forms.Select(attrs={'class': 'blue-input-box'}),
    )
=======
# forms.py
from django import forms
from .models import Statement, Enforcer, Case, Witness

class StatementForm(forms.ModelForm):
    class Meta:
        model = Statement
        fields = ['enforcer_id', 'case_description', 'crime_location', 'crime_victim', 'crime_incident_date', 'crime_incident_time']


from django import forms
from .models import WitnessTestification

class WitnessTestificationForm(forms.ModelForm):
    class Meta:
        model = WitnessTestification
        fields = ['witness_id', 'case_description', 'case_location', 'case_victim', 'case_incident_date', 'case_incident_time', 'case_potential_suspect']


# forms.py
from django import forms
from .models import SuspectTestification

class SuspectTestificationForm(forms.ModelForm):
    class Meta:
        model = SuspectTestification
        fields = ['suspect_id', 'case_description', 'crime_location', 'crime_victim', 'crime_incident_date', 'crime_incident_time']
>>>>>>> 7ff2cbcfdbc7181a4fe291243adbdb7c4db75281
