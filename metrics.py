from keras import backend as K


# https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/losses/jaccard.py
def jaccard_distance(y_true, y_pred, smooth=100):
  intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
  sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
  jac = (intersection + smooth) / (sum_ - intersection + smooth)
  return (1 - jac) * smooth