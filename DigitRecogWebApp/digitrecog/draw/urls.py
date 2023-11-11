from django.urls import path
from . import views


app_name = 'draw'

urlpatterns = [
    path('', views.draw_view, name='draw'),
    path('predict', views.predict_view, name='predict'),
    path('update', views.update_model, name='update')
]
