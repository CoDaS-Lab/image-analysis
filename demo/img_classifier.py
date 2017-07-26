import os
import sys
from sklearn import datasets, metrics
from image_analysis.pipeline import Pipeline
from image_analysis.pipeline import SVM

digits = datasets.load_digits()
images_and_labels = list(zip(digits.images, digits.target))
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

pipe = Pipeline(models={'SVM': SVM()})
pipe.train(data[:n_samples // 2], digits.target[:n_samples // 2])

expected = digits.target[n_samples // 2:]
predicted = pipe.predict(data[n_samples // 2:])
print(metrics.classification_report(expected, predicted['SVM']))
