from django.urls import path
from .views import *
urlpatterns = [
    path('', DepLoginView.as_view(), name='index'),
    path('home', HomePageView.as_view(), name='home'),
    path('error/', ErrorPageView.as_view(), name='error'),
    path('success/<str:serial_number>/', SuccessPageView.as_view(), name='success'),
    path('login/', LoginView.as_view(), name='login'),
    path('forms/', FormsView.as_view(), name='forms'),
    path('signup/', signup, name='signup'),
    path('depsignup/', depsignup, name='depsignup'),
    path('logout/', logout, name='logout'),
    path('reset-password/', reset_password, name='reset_password'),
    path('reset-password/<str:token>/', reset_password_confirm, name='reset_password_confirm'),
    ]