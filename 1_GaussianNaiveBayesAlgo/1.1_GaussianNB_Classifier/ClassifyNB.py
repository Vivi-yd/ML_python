### Solution of Problem by Vivi

def classify(features_train, labels_train):   
    ### import the sklearn module for GaussianNB
    from sklearn.naive_bayes import GaussianNB
    ### create classifier 
    clf = GaussianNB()
    ### fit the classifier on the training features and labels
    ### return the fit classifier
      
    fit = clf.fit(features_train, labels_train)
    
    return fit