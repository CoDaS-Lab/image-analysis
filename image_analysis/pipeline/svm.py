# Copyright 2017 Codas Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from sklearn import svm
from image_analysis.pipeline.feature import Feature


class SVM(Feature):
    """
    DESCRIPTITON:
        scikit-learn Support vector machine wrapper

    PARAMS:
        :gamma: kernel coefficient
    """

    def __init__(self, gamma=.001):
        Feature.__init__(self, 'Support Vector Machine')
        self.classifier = svm.SVC(gamma=gamma)

    def train(self, X, y):
        """
        DESCRIPTION:
            train the svm

        PARAMS:
            :X: the data
            :y: the labels for the data
        """
        self.classifier.fit(X, y)

    def predict(self, X):
        """
        DESCRIPTION:
            predrict values for new data

        PARAMS:
            :X: data to predict labels for
        """
        return self.classifier.predict(X)
