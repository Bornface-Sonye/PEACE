<<<<<<< HEAD
from django.urls import path
from .views import HomePageView,InterrogatorDashboardView, AddAnswerView, InterrogatorReportView, ErrorPageView, SuccessPageView
urlpatterns = [
    path('interrogator/', HomePageView.as_view(), name='index'),
    path('interrogator/dashboard/', InterrogatorDashboardView.as_view(), name='dashboard'),
    path('interrogator/dashboard/answer/', AddAnswerView.as_view(), name='answer'),
    path('interrogator/dashboard/report/', InterrogatorReportView.as_view(), name='report'),
    path('error/', ErrorPageView.as_view(), name='error'),
    path('success/', SuccessPageView.as_view(), name='success'),
    ]
=======
from django.urls import path
from . import views
from .views import statement_form, success_page, submit_testification, add_suspect_testification
from django.urls import path
from .views import data_fusion
from .views import load_report

urlpatterns = [
    path('', views.home, name='index'),
    path('statement/', statement_form, name='statement_form'),
    path('success/', success_page, name='success_page'),
    path('testify/', submit_testification, name='submit_testification'),
    path('testifys/', add_suspect_testification, name='add_suspect_testification'),
    path('report/', views.load_report, name='load_report'),
    path('fusion/', data_fusion, name='data_fusion'),
]

>>>>>>> 7ff2cbcfdbc7181a4fe291243adbdb7c4db75281
