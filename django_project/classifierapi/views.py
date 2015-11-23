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


def doPCA(data, n=2):
        from sklearn.decomposition import PCA
        pca = PCA(n_components = n)
        pca.fit(data)
        return pca

def plot(clf, X_test, y_test):
    #########courtesy of Udacity######################
    x_min = 0.0; x_max = 1.0
    y_min = 0.0; y_max = 1.0

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    h = .01  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    #plt.pcolormesh(xx, yy, Z, cmap=pl.cm.seismic)

    # Plot also the test points
    pc_1 = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==0]
    pc_2 = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==0]
    pc_1_pos = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==1]
    pc_2_pos = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==1]

    plt.scatter(pc_1, pc_2, color = "b", label="neg")
    plt.scatter(pc_1_pos, pc_2_pos, color = "r", label="pos")
    plt.legend()
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.savefig("test.png")
           

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
        accuracy = metrics.accuracy_score(labels_test, pred)
        print "accuracy: ", accuracy
        request.session['accuracy'] = accuracy
        confusion_matrix = metrics.confusion_matrix(labels_test, pred)
        print "confusion matrix: ", confusion_matrix
        request.session['confusion_matrix'] = confusion_matrix.tolist()
        f1 = metrics.f1_score(labels_test, pred)
        request.session['f1'] = f1
        precision = metrics.precision_score(labels_test, pred)
        request.session['precision'] = precision
        recall = metrics.recall_score(labels_test, pred)
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





