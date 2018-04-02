from sklearn import metrics
import numpy as np

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
    
    
    

def to_one_hot(y):
    y = np.array(y, dtype=int)
    n_values = int(np.max(y)) + 1
    y = np.eye(n_values)[y]
    return y

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    np.random.seed(172)
    p = np.random.permutation(len(a))
    return a[p], b[p]


#
#
#
# from tensorflow.python.framework import tensor_shape
#
# def get2d_deconv_output_size(input_height, input_width, filter_height,
#                            filter_width, row_stride, col_stride, padding_type):
#     """Returns the number of rows and columns in a convolution/pooling output."""
#     input_height = tensor_shape.as_dimension(input_height)
#     input_width = tensor_shape.as_dimension(input_width)
#     filter_height = tensor_shape.as_dimension(filter_height)
#     filter_width = tensor_shape.as_dimension(filter_width)
#     row_stride = int(row_stride)
#     col_stride = int(col_stride)
#
#     # Compute number of rows in the output, based on the padding.
#     if input_height.value is None or filter_height.value is None:
#       out_rows = None
#     elif padding_type == "VALID":
#       out_rows = (input_height.value - 1) * row_stride + filter_height.value
#     elif padding_type == "SAME":
#       out_rows = input_height.value * row_stride
#     else:
#       raise ValueError("Invalid value for padding: %r" % padding_type)
#
#     # Compute number of columns in the output, based on the padding.
#     if input_width.value is None or filter_width.value is None:
#       out_cols = None
#     elif padding_type == "VALID":
#       out_cols = (input_width.value - 1) * col_stride + filter_width.value
#     elif padding_type == "SAME":
#       out_cols = input_width.value * col_stride
#
#     return out_rows, out_cols
#
# out_rows, out_cols = get2d_deconv_output_size(input_height=32, input_width=32, filter_height=3,
#                            filter_width=3, row_stride=1, col_stride=1, padding_type='VALID')
# print (out_rows, out_cols )