from django.urls import path
from .views import *
urlpatterns = [
    path('', HomePageView.as_view(), name='index'),
    path('interrogator/dashboard/', InterrogatorDashboardView.as_view(), name='main_dashboard'),
    path('dashboard/', DashboardView.as_view(), name='dashboard'),
    #path('interrogator/dashboard/answer/', AddAnswerView.as_view(), name='answer'),
    #path('interrogator/dashboard/report/', InterrogatorReportView.as_view(), name='report'),
    path('error/', ErrorPageView.as_view(), name='error'),
    path('success/<str:serial_number>/', SuccessPageView.as_view(), name='success'),
    path('login/', LoginView.as_view(), name='login'),
    path('forms/', FormsView.as_view(), name='forms'),
    path('signup/', signup, name='signup'),
    path('deplogin/', DepLoginView.as_view(), name='deplogin'),
    path('depsignup/', depsignup, name='depsignup'),
    path('logout/', logout, name='logout'),
    path('reset-password/', views.reset_password, name='reset_password'),
    path('reset-password/<str:token>/', views.reset_password_confirm, name='reset_password_confirm'),
    #path('generate-pdf-report/', GeneratePDFReportView.as_view(), name='generate_pdf_report'),
    ]

# In your urls.py
urlpatterns = [
    # Other URLs
    
]
