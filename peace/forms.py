from django import forms
from .models import Enforcer

class EnforcerSignUpForm(forms.ModelForm):
    confirm_password = forms.CharField(max_length=100, widget=forms.PasswordInput())

    class Meta:
        model = Enforcer
        fields = ['enforcer_email', 'first_name', 'last_name', 'password']
        widgets = {
            'password': forms.PasswordInput(),
        }


from django import forms
from django.contrib.auth.forms import AuthenticationForm

class EnforcerLoginForm(AuthenticationForm):
    pass


from django import forms
from .models import FirstQuestioning, SecondQuestioning

class FirstQuestioningForm(forms.ModelForm):
    class Meta:
        model = FirstQuestioning
        fields = ['suspect_email', 'case', 'question1_answer', 'question2_answer', 'question3_answer']

class SecondQuestioningForm(forms.ModelForm):
    class Meta:
        model = SecondQuestioning
        fields = ['suspect_email', 'case', 'question1_answer', 'question2_answer', 'question3_answer']


