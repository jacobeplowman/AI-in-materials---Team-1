import statistics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def feature_importances_rf_reg(X,y,estimators,test_size):
    feature_names=X.columns.values
    X_arr=np.array(X)
    scores =[]
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X_arr, y, test_size=test_size)
        rf = RandomForestRegressor(n_estimators=estimators)
        rf.fit(X_train, y_train)
        scores.append(rf.feature_importances_)
    scores=np.array(scores)
    new_scores=[]
    stdevs=[]
    for i in range(len(X_arr[0])):
        new_scores.append(statistics.mean(scores[:,i]))
        stdevs.append(statistics.stdev(scores[:,i]))
        
    importances=[]
    for name, score, stdev in zip(feature_names, new_scores, stdevs):
        importances.append((name, score, stdev))
    
    fi = sorted(importances, key = lambda x: x[1],reverse=True)[:10]
    
    xx=[]
    yy=[]
    ee=[]
    
    for i in fi:
        #xx.append(str(i[0])[2:-3])
        xx.append(str(i[0]))
        
        yy.append(i[1])
        ee.append(i[2])
        
    xx2=[]
    yy2=[]
    ee2=[]
    

              
    
    plt.figure(figsize=(30,20))
    plt.bar(xx, yy)
    plt.xlabel("Features")
    plt.ylabel("Relative importance")
    plt.errorbar(xx, yy, yerr=ee, fmt="o", color="r")
    plt.show()
    
    
    for i in list(range(len(xx))):
        
        X1=np.array(X[xx[i]]).reshape(-1,1)
        
        lin_reg = LinearRegression()
        lin_reg.fit(X1,y)
        y_pred=lin_reg.predict(X1)
        
        score=lin_reg.score(X1,y)
        RMSE=np.sqrt(mean_squared_error(X1,y_pred))

        plt.figure(figsize=(10,5))
        plt.plot(X[xx[i]], y_pred, 'r-', linewidth = 0.5, label = 'r squared =' + str(score))
        plt.legend(loc = 'upper center', fontsize = 14)
        plt.scatter(X[xx[i]],y,s=1)
        plt.title('Linear regression of ' + str((xx[i]))+' vs '+'band gap' , fontsize = 16)
        plt.xlabel((xx[i]), fontsize = 16)
        plt.ylabel('band gap (eV)', fontsize = 16)
        plt.show()