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
from django.template import RequestContext, loader
import matplotlib.pyplot as plt
import matplotlib 
import pylab as pl
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
    template = loader.get_template('results.html')
    file_path = request.session['file_path']
    print file_path
    context = RequestContext(request, {'test': file_path})
    
    return HttpResponse(template.render(context))

class LogisticRegression(APIView):

    def get(self, request):
      
        return HttpResponse("Stub")


    def post(self, request):

        #retrieve info from request
        client_info = request.body
       
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
        sensor_array = params["sensor_array"] 
        request.session['sensors'] = sensor_array
        fit_using_pca = params["fit_using_pca"]
        request.session['fit_using_pca'] = fit_using_pca
        delta = jobj['delta']

        #make sure the number of principle components is at least 2
        no_pc = params["no_pc"]
        request.session['no_pc'] = no_pc
        if no_pc < 2: 
            no_pc = 2                
       
        data_array = np.asarray([])

        for column in sensor_array:
            column_list = np.asarray([d[column] for d in data]).astype(np.float32)
           
            if data_array.size > 0:
                data_array = np.column_stack((data_array, column_list))
            else:
                data_array = column_list
     
        y = np.array([d['in_category'] for d in data]).astype(np.float32)       

        # split data into training and test sets
        features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(
            data_array, y, test_size=0.4, random_state=0)


        # get logistic regression classifier
        lg = lm.LogisticRegression()

        #tune parameters using GridSearchCV
        parameters = {'C': C_array}
        request.session['c_params'] = C_array


        grdlog = grid_search.GridSearchCV(lg, parameters)

        # train logreg    
        if(fit_using_pca):
            pca = doPCA(features_train, no_pc)
            transformed_features_train = pca.transform(features_train)
            transformed_labels_train = pca.transform(labels_train)
            grdlog.fit(transformed_features_train, transformed_labels_train)
            request.session['explained_variance_ratio', pca.explained_variance_ratio]

        else:            
            grdlog.fit(features_train, labels_train)
            lg.set_params(**grdlog.best_params_)
            lg.fit(features_train, labels_train)
            pred = lg.predict(features_test)
            accuracy = metrics.accuracy_score(pred, labels_test)
            request.session['accuracy'] = accuracy

          
        
        request.session['best_parameters'] = grdlog.best_params_

        request.session['params'] = lg.coef_.tolist()

        request.session['y_intercept'] = lg.intercept_.tolist()
     

        ### to plot the data with decision boundary
        #generate decision boundary with top 2 principle components
        # graph_pca = doPCA(features_train, 2)
        # graph_transformed_features = graph_pca.transform(features_train)
        # #labels_pca = doPCA(labels_train, 2)

        # #graph_transformed_labels = labels_pca.transform(labels_train)
        # graph_transformed_labels = labels_train

        # log_p = lm.LogisticRegression()
        # logreg_p = grid_search.GridSearchCV(log_p, parameters)

        # logreg_p.fit(graph_transformed_features, graph_transformed_labels)
        # log_p.set_params(**logreg_p.best_params_)

        
        #plot
        #plot(logreg_p, graph_transformed_features, graph_transformed_labels)  
        
        request.session['file_path'] = "data.txt"

        result = {'name': name, 'category': category, 
        'method': method,'params': lg.coef_[0].tolist(), 
        'intercept': str(lg.intercept_[0]), 'sensors': sensor_array, 'delta': delta,
        'accuracy': accuracy}
      
        return HttpResponse(json.dumps(result))




