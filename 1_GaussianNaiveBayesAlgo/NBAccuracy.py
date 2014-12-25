def NBAccuracy(features_train, labels_train, features_test, labels_test):
    
    """ compute the accuracy of your Naive Bayes classifier """
    ### import the sklearn module for GaussianNB
    from sklearn.naive_bayes import GaussianNB

    ### create classifier (like defining a formula)
    clf = GaussianNB()

    ### fit the classifier on the training features and labels (like training it with data to get correct parameters)
    clf.fit(features_train, labels_train)

    ### use the trained classifier to predict labels for the test features (like using the formula to predict with new data input)
    pred = clf.predict(features_test)
    
    
    ### import an sklearn module for calculating accuracy of the classifier
    from sklearn.metrics import accuracy_score
    ### calculate and return the accuracy on the test data
    accuracy = accuracy_score(labels_test, pred)
    return accuracy
