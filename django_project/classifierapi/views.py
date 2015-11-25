from rest_framework.views import APIView
from django.http import HttpResponse
import numpy as np
np.set_printoptions(threshold=np.nan)
import sklearn.linear_model as lm
from sklearn import cross_validation, grid_search, metrics
import json
import pickle
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.template import RequestContext, Context, loader, Template
import matplotlib.pyplot as plt
import matplotlib 
import pylab as pl
import re

matplotlib.use('agg')

def index(request):
    return HttpResponse("hello world")

def results(request): 
    return render(request, 'results.html', request.session)

class LogisticRegression(APIView):

    def get(self, request):
      
        return HttpResponse("Stub")


    def post(self, request):
        #retrieve info from request
        print "request received"
        client_info = request.body
        print "loading data"
        jobj = json.loads(client_info)      

        data = jobj['data']
        name = jobj['name']

        request.session['name'] = name
        category = jobj['class']
        request.session['class'] = category
        method = jobj['method']
        request.session['method'] = method
        params = json.loads(jobj['params'])
        C_array = params['C_array']
        request.session['C_array'] = C_array
        sensor_array = params["sensor_array"] 
        request.session['sensors'] = sensor_array
                      
        data_array = np.asarray([])

        for column in sensor_array:
            column_list = np.asarray([d[column] for d in data]).astype(np.float32)
           
            if data_array.size > 0:
                data_array = np.column_stack((data_array, column_list))
            else:
                data_array = column_list
     
        y = np.array([d['in_category'] for d in data]).astype(np.float32) 
        
        print "splitting data into training and test sets"
        # split data into training and test sets
        features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(
            data_array, y, test_size=0.4, random_state=0)

        # get logistic regression classifier
        print "creating classifier"
        lg = lm.LogisticRegression()

        #tune parameters using GridSearchCV
        print "tuning parameters for regression"
        parameters = {'C': C_array}
        print "parameters: ", parameters
        request.session['c_params'] = C_array

        grdlog = grid_search.GridSearchCV(lg, parameters)
        
        
        # train 
        print "training data"             
        grdlog.fit(features_train, labels_train)
        lg.set_params(**grdlog.best_params_)
        print "fitting data"
        lg.fit(features_train, labels_train)
        print "generating predictions using test set"
        pred = lg.predict(features_test)
        print "getting metrics for predicions"
        accuracy = metrics.accuracy_score(pred, labels_test)
        print "accuracy: ", accuracy
        request.session['accuracy'] = accuracy
        confusion_matrix = metrics.confusion_matrix(pred, labels_test)
        print "confusion matrix: ", confusion_matrix
        request.session['confusion_matrix'] = confusion_matrix.tolist()
        f1 = metrics.f1_score(pred, pred)
        request.session['f1'] = f1
        precision = metrics.precision_score(pred, labels_test)
        request.session['precision'] = precision
        recall = metrics.recall_score(pred, labels_test)
        request.session['recall'] = recall               
        
        print "getting parameters"
        parameters = lg.coef_.tolist()[0]
        sensor_params = zip(sensor_array, parameters)
        request.session["sensor_params"] = sensor_params
        request.session['y_intercept'] = lg.intercept_.tolist()[0]
        request.session['best_parameters'] = grdlog.best_params_['C']
        
        print "sending result"  
        result = {"name": name, "category": category, 
        "method": method,"params": lg.coef_[0].tolist(), 
        "intercept": str(lg.intercept_[0]), "sensors": sensor_array, 
        "accuracy": accuracy}

        request.session["results"] = str(result)

        html = str(render(request, "results.html", request.session))    
        response = {"html": html, "results": result}
        json_response = json.dumps(response)

        return HttpResponse(json_response)





