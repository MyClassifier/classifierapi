from django.conf.urls import url

from . import views
from views import LogisticRegression

urlpatterns = [
    url(r'^logistic_regression', LogisticRegression.as_view(), 
    	name='logistic_regression'),
    
]

