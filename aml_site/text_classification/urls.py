from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^submit_text$', views.submit_text, name='submit_text'),
    url(r'^send_email$', views.send_feedback_email, name='send_email')
]
