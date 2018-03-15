from sklearn import metrics

class Score():
    '''
    Only provide binary input
    Scores are calculated based on 1
    '''
    
    @staticmethod
    def recall(y_true, y_pred, reverse=False):
        ''' tp / (tp+fn) --> Accuracy(y_true = 1 | y_pred = 1) '''
        if reverse:
            y_true = 1 - y_true
            y_pred = 1 - y_pred
        return metrics.recall_score(y_true, y_pred)
    
    @staticmethod
    def precision(y_true, y_pred, reverse=False):
        ''' tp / (tp+fp) '''
        if reverse:
            y_true = 1 - y_true
            y_pred = 1 - y_pred
        return metrics.precision_score(y_true, y_pred)
    
    @staticmethod
    def accuracy(y_true, y_pred):
        return metrics.accuracy_score(y_true, y_pred)
    
    @staticmethod
    def auc(y_true, y_pred):
        return metrics.roc_auc_score(y_true, y_pred)