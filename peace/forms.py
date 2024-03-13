from django import forms
from .models import Suspect, Case, SuspectResponse
from .models import Enforcer

class SignUpForm(forms.ModelForm):
    confirm_password = forms.CharField(widget=forms.PasswordInput)

    class Meta:
        model = Enforcer
        fields = ['enforcer_email', 'badgeNo', 'password_hash']  # Update to use 'password_hash' field

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get("password_hash")  # Update to use 'password_hash' field
        confirm_password = cleaned_data.get("confirm_password")

        if password != confirm_password:
            raise forms.ValidationError(
                "Password and confirm password do not match"
            )

    def save(self, commit=True):
        instance = super().save(commit=False)
        instance.set_password(self.cleaned_data["password_hash"])  # Update to use 'password_hash' field
        if commit:
            instance.save()
        return instance


class LoginForm(forms.Form):
    enforcer_email = forms.EmailField()
    password = forms.CharField(widget=forms.PasswordInput)

class AnswerForm(forms.ModelForm):
    case_description = forms.ModelChoiceField(
        queryset=Case.objects.all(),
        required=True,
        label='Case Description',
        widget=forms.Select(attrs={'class': 'blue-input-box'}),
    )
    
    
    suspect_email = forms.ModelChoiceField(
        queryset=Suspect.objects.all(),
        required=True,
        label='Suspect Email Address',
        widget=forms.Select(attrs={'class': 'blue-input-box'}),
    )
    
    
    suspect_Residence_county = forms.ModelChoiceField(
        queryset=County.objects.all(),
        required=True,
        label='Suspect County of Residence: ',
        widget=forms.Select(attrs={'class': 'blue-input-box'}),
    )
    
    incident_county = forms.ModelChoiceField(
        queryset=County.objects.all(),
        required=True,
        label='County Of the Incident: ',
        widget=forms.Select(attrs={'class': 'blue-input-box'}),
    )
    
    class Meta:
        model = SuspectResponse
        fields = ['case_description', 'suspect_email', 'suspect_Residence_county', 'incident_county', 'trace', 'know_complainant', 'involved_with_complainant', 'recidivist']
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
        max_length=8,
        help_text="Enter the 8-alphanumeric serial number",
        widget=forms.TextInput(attrs={'class': 'blue-input-box'})
    )
