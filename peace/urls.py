from django.urls import path
from .views import *
from .views import ErrorPageView, SuccessPageView
from .views import HomePageView,InterrogatorDashboardView, AddAnswerView, InterrogatorReportView
urlpatterns = [
    path('', HomePageView.as_view(), name='index'),
    path('interrogator/dashboard/', InterrogatorDashboardView.as_view(), name='dashboard'),
    path('interrogator/dashboard/answer/', AddAnswerView.as_view(), name='answer'),
    path('interrogator/dashboard/news/', AddNewsView.as_view(), name='news'),
    path('interrogator/dashboard/feedback/', AddFeedbackView.as_view(), name='feedback'),
    path('interrogator/dashboard/report/', InterrogatorReportView.as_view(), name='report'),
    path('error/', ErrorPageView.as_view(), name='error'),
    path('success/', SuccessPageView.as_view(), name='success'),
    path('login/', login, name='login'),
    path('signup/', signup, name='signup'),
    path('deplogin/', deplogin, name='deplogin'),
    path('depsignup/', depsignup, name='depsignup'),
    ]
