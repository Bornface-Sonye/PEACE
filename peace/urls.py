from django.urls import path
from .views import enforcer_signup
from .views import enforcer_login
from .views import first_questioning, second_questioning, success

urlpatterns = [
    path('enforcer/signup/', enforcer_signup, name='enforcer_signup'),
    path('enforcer/login/', enforcer_login, name='enforcer_login'),
    path('first_questioning/', first_questioning, name='first_questioning'),
    path('second_questioning/', second_questioning, name='second_questioning'),
    path('success/', success, name='success'),
]
