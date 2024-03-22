from django import forms
from .models import Suspect, Case, SuspectResponse, Department, BadgeNumber
from .models import Enforcer, CaseCollection, EnforcerCase, SuspectCase, County

from django import forms
from .models import BadgeNumber, Enforcer, DepartmentNumber

from django import forms
from .models import Enforcer, BadgeNumber

class SignUpForm(forms.ModelForm):
    confirm_password = forms.CharField(widget=forms.PasswordInput)

    class Meta:
        model = Enforcer
        fields = ['badge_no', 'password']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['badge_no'] = forms.ModelChoiceField(
            queryset=BadgeNumber.objects.all(),
            required=True,
            label='Badge Number:',
            widget=forms.Select(attrs={'class': 'black-input-box'})
        )

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get("password")
        confirm_password = cleaned_data.get("confirm_password")

        if password != confirm_password:
            raise forms.ValidationError(
                "Password and confirm password do not match"
            )

    def save(self, commit=True):
        instance = super().save(commit=False)
        instance.set_password(self.cleaned_data["password"])
        if commit:
            instance.save()
        return instance
    
from django import forms

class LoginForm(forms.Form):
    badge_no = forms.DecimalField(label='Badge Number')
    password = forms.CharField(label='Password', widget=forms.PasswordInput)
    
class DepSignUpForm(forms.ModelForm):
    confirm_password = forms.CharField(widget=forms.PasswordInput)

    class Meta:
        model = Department
        fields = ['dep_no', 'password']  # Update to use 'password_hash' field
        
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['dep_no'] = forms.ModelChoiceField(
            queryset=DepartmentNumber.objects.all(),
            required=True,
            label='Department Number:',
            widget=forms.Select(attrs={'class': 'black-input-box'})
        )

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get("password")  # Update to use 'password_hash' field
        confirm_password = cleaned_data.get("confirm_password")

        if password != confirm_password:
            raise forms.ValidationError(
                "Password and confirm password do not match"
            )

    def save(self, commit=True):
        instance = super().save(commit=False)
        instance.set_password(self.cleaned_data["password"])  # Update to use 'password_hash' field
        if commit:
            instance.save()
        return instance

class DepLoginForm(forms.Form):
    dep_no = forms.CharField(label='Department Number')
    password = forms.CharField(label='Password', widget=forms.PasswordInput)
    

class AnswerForm(forms.ModelForm):
    case_description = forms.ModelChoiceField(
        queryset=Case.objects.all(),
        required=True,
        label='Case Description',
        widget=forms.Select(attrs={'class': 'blue-input-box'}),
    )
    
    
    unique_id = forms.ModelChoiceField(
        queryset=Suspect.objects.all(),
        required=True,
        label='Suspect Unique Identity:',
        widget=forms.Select(attrs={'class': 'blue-input-box'}),
    )
    
    
    class Meta:
        model = SuspectResponse
        fields = ['case_description', 'unique_id', 'trace', 'know_complainant', 'involved_with_complainant', 'recidivist']
        labels = {
            'trace': 'Is there a trace of Suspect in Crime Scene ? ',
            'know_complainant': 'Do the Suspect know the person filing the complaint ? ',
            'involved_with_complainant': 'Have the Suspect involved in any quarrel  with complainant ? ',
            'recidivist': 'Have the Suspect been involved in a similar case before ? ',
        }
        widgets = {
            'trace': forms.Select(choices=[('Yes', 'Yes'), ('No', 'No')], attrs={'class': 'black-input-box'}),
            'know_complainant': forms.Select(choices=[('Yes', 'Yes'), ('No', 'No')], attrs={'class': 'black-input-box'}),
            'involved_with_complainant': forms.Select(choices=[('Yes', 'Yes'), ('No', 'No')], attrs={'class': 'black-input-box'}),
            'recidivist': forms.Select(choices=[('Yes', 'Yes'), ('No', 'No')], attrs={'class': 'black-input-box'}),
        }

class InterrogatorReportForm(forms.Form):
    serial_number = forms.CharField(
        max_length=50,
        help_text = "Enter the 50-alphanumeric serial number",
        widget=forms.TextInput(attrs={'class': 'blue-input-box'})
    )
