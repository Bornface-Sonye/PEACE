from django.shortcuts import render, redirect
from .forms import EnforcerSignUpForm

def enforcer_signup(request):
    if request.method == 'POST':
        form = EnforcerSignUpForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')  # Redirect to login page after successful signup
    else:
        form = EnforcerSignUpForm()
    return render(request, 'enforcer_signup.html', {'form': form})



from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from .forms import EnforcerLoginForm

def enforcer_login(request):
    if request.method == 'POST':
        form = EnforcerLoginForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('dashboard')  # Redirect to enforcer dashboard after successful login
    else:
        form = EnforcerLoginForm()
    return render(request, 'enforcer_login.html', {'form': form})


from django.shortcuts import render, redirect
from .forms import FirstQuestioningForm, SecondQuestioningForm

def first_questioning(request):
    enforcer = request.user.enforcer
    cases = Case.objects.filter(enforcer=enforcer)
    if request.method == 'POST':
        form = FirstQuestioningForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('second_questioning')
    else:
        form = FirstQuestioningForm()
    return render(request, 'first_questioning.html', {'form': form, 'cases': cases})

def second_questioning(request):
    if request.method == 'POST':
        form = SecondQuestioningForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('success')
    else:
        form = SecondQuestioningForm()
    return render(request, 'second_questioning.html', {'form': form})

def success(request):
    return render(request, 'success.html')
