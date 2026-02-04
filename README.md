## XgBoost CNN 

- dataset- cifar10
- Results
    - CNN model accuracy - 62 %
    - XgBoost model accuracy (inputs taken from feature layer of CNN model) - 70 %

- Todo
    - Test XgBoost model performance on cifar dataset (i.e no CNN model input).
    - use proper metrics(log loss, precision, recall, f1, ROC curve) to check how "good" the model is on classifying the images?
    - instead of custom CNN model, apply VGG/resnet architecture on dataset and then check how the xgboost model performance is on those conv features.
    - apart from xgboost, try out SVM, KNN.
