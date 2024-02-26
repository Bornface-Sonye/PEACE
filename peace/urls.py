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
