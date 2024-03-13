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
    
    class Meta:
        model = SuspectResponse
        fields = ['case_description', 'suspect_email', 'trace', 'know_complainant', 'reason_to_lie', 'other_complaints', 'involved_with_complainant', 'involved_in_similar_case']
        labels = {
            'trace': 'Are there any traces of the suspect in crime scene ?',
            'know_complainant': 'Do you know the person filing the complaint ?',
            'reason_to_lie': 'Can you think of any reason why someone would lie about this incident ?',
            'involved_with_complainant': 'Have you involved in any row or quarrel  with complainant ?',
            'involved_in_similar_case': 'Have you been involved in a similar case before ?',
        }
        widgets = {
            'trace': forms.TextInput(attrs={'class': 'black-input-box'}),
            'know_complainant': forms.Select(choices=[('Yes', 'Yes'), ('No', 'No')], attrs={'class': 'black-input-box'}),
            'reason_to_lie': forms.TextInput(attrs={'class': 'black-input-box'}),
            'involved_with_complainant': forms.Select(choices=[('Yes', 'Yes'), ('No', 'No')], attrs={'class': 'black-input-box'}),
            'involved_in_similar_case': forms.Select(choices=[('Yes', 'Yes'), ('No', 'No')], attrs={'class': 'black-input-box'}),
        }
      
class InterrogatorReportForm(forms.Form):
    serial_number = forms.CharField(
        max_length=8,
        help_text="Enter the 8-alphanumeric serial number",
        widget=forms.TextInput(attrs={'class': 'blue-input-box'})
    )
