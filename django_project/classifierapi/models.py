from django.db import models

# Create your models here.
class User(models.Model):
	user_name = models.CharField(max_length=200)
	password = models.CharField(max_length=200)


class Model(models.Model):
	user = models.ForeignKey(User)
	model_file = models.CharField(max_length=2000)
	
