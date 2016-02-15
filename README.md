# classifierapi
This is the second component in MyClassifier: It's a Django RESTful service that: takes labeled sensor data from dataGatherer in the form of JSON, runs logistic regression on the data, and returns a set of trained parameters to the phone. 

The api currently lives on Digital Ocean (email me to for more information).

This is a working prototype: refactoring in process. 

More documentation coming soon!



References/Acknowledgements
Many thanks to Udacity's Machine Learning Course (https://www.udacity.com/course/intro-to-machine-learning--ud120) with Sabastian Thrun and Katie Malone, some small pieces of code come from this course, as well as the general approach for using Scikit-learn.  

Scikit-learn: the best ML library in the world, in my humble opinion: http://scikit-learn.org/stable/
Also to stackoverflow.com, for many tutorials and useful code bits.
Django: my favorite web framework: https://www.djangoproject.com/
Django Rest Framework: makes it easy to create a REST api with django: http://www.django-rest-framework.org/
