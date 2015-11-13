from rest_framework.views import APIView
from .mixins import JSONResponseMixin
from django.http import HttpResponse
import numpy as np
import sklearn.linear_model as lm
from sklearn import cross_validation
import json
import pickle
from django.shortcuts import render
from django.http import HttpResponse

def index(request):
    return HttpResponse("Hello, world! This is our first view.")



class LogisticRegression(APIView):

    def get(self, request):
      
        return HttpResponse("Stub")


    def post(self, request):

        #retrieve info from request
        client_info = request.body

        jobj = json.loads(client_info)

        data = jobj['data']
        name = jobj['name']
        category = jobj['class']
        method = jobj['method']
        params = json.loads(jobj['params'])

        C_array = params['C_array']
        sensor_array = params["sensor_array"]
        fit_using_pca = params["fit_using_pca"]
        no_pc = params["no_pc"]

        # print "name: %s, category: %s, method: %s " % (name, category, method)

        print "C_array: ", C_array
        print "sensor_array: ", sensor_array
        print "fit_using_pca: ", fit_using_pca
        print "no_pc: ", no_pc
        
        # get data for chosen features






    # def post(self, request):
    #     array =  '{"data":' + request.body + '}'
    #     #print array
    #     jobj= json.loads(array)
   
    #     jarray = jobj['data']
    #     matrix = np.asarray([[j['GravityX'], j['GravityY'], j['GravityZ'], j['true']] for j in jarray])

    #     X = matrix[:, :3]
    #     Y = matrix[:, 3]
    
    #     clf = svm.SVC()
    #     result = clf.fit(X, Y)
    #     print result

        
        #matrix = np.asarray([[j['GravityX'], j['GravityY'], j['GravityZ'], j['true']] for j in jarray])

        # X = matrix[:, :3]
        # y = matrix[:, 3]
        # print y
        # logreg = lm.LogisticRegression()
      
        # logreg.fit(X, y) 
        # print logreg.coef_     
        # #sigmoid( dot([val1, val2], lr.coef_) + lr.intercept_ ) for prediction 

        return HttpResponse( json.dumps(jobj) )


#         {
#     "name": "h",
#     "class": "y",
#     "method": "log_reg",
#     "params": {
#     "C_array": [
#         0.01,
#         0.1,
#         1,
#         10,
#         100,
#         0.01,
#         0.1,
#         1,
#         10,
#         100
#     ],
#     "sensor_array": [
#         "accelerationX",
#         "accelerationY",
#         "accelerationZ",
#         "gyroX",
#         "gyroY",
#         "gyroZ",
#         "magneticX",
#         "magneticY",
#         "magneticZ",
#         "delta_accelX",
#         "delta_accelY",
#         "delta_accelZ",
#         "delta_gyroX",
#         "delta_gyroY",
#         "delta_gyroZ",
#         "delta_magneticX",
#         "delta_magneticY",
#         "delta_magneticZ"
#     ],
#     "fit_using_pca": false,
#     "no_pc": 0
# }
#     "delta": 0
# }

