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


import numpy as np
import copy
from image_analysis.pipeline.feature import Feature


class Pipeline:
    """
    DESCRIPTION:
        This class is the bridge between running features on images.
        Automates the process of extracting each feature, saving it,
        outputing it and putting it in dictionary form

    PARAMS:
        :data: image data
        :ops: features to run on images. These ops don't have dependencies
        :seq: features to run in sequential way (output is input to another)
        :save_all: boolean check to save all features ran
        :models: dictionary of statistical models to run on the data
    """
    def __init__(self, data=None, ops=None, seq=None, save_all=None,
                 models=None):
        self.data = data
        self.models = models
        self.save_all = save_all

        self.batch_ops = None
        self.frame_ops = None
        self.seq_ops = None
        self.set_ops(ops, seq)

        self.empty_frame = {}           # Defined by set method below.
        self.output = []          # Output data structure of the pipeline.

    def set_ops(self, ops=None, seq=None):
        """
        DESCRIPTION:
            puts ops and seq into separate list for better managing

        PARAMS:
            :ops: features to run on images. These ops don't have dependencies
            :seq: features to run in sequential way
                  (output is input to another)
        """
        batch_ops = []
        frame_ops = []
        if ops is not None:
            for op in ops:
                assert isinstance(op, Feature)
                if op.batch_op:
                    batch_ops.append(op)
                elif op.frame_op:
                    frame_ops.append(op)
                else:
                    raise ValueError('One and only one of either batch op or' +
                                     'or frame op is allowed to be True!')
        elif ops is None or ops == []:
            batch_ops = []
            frame_ops = []

        if seq is not None:
            for op in seq:
                assert isinstance(op, Feature)
        elif seq is None or seq == []:
            seq = []

        self.batch_ops = batch_ops
        self.frame_ops = frame_ops
        self.seq_ops = seq

    def set_batch_ops(self, batch_ops=None):
        """
        DESCRIPTION:
            update batch operations list

        PARAMS:
            :batch_ops: features to put in batch_ops
        """
        if batch_ops is None:
            batch_ops = []

        for op in batch_ops:
            assert isinstance(op, Feature)
            assert op.batch_op

        self.batch_ops = batch_ops

    def set_frame_ops(self, frame_ops=None):
        """
        DESCRIPTION:
            update frame operations list

        PARAMS:
            :frame_ops: features to put in frame_ops
        """
        if frame_ops is None:
            frame_ops = []

        for op in frame_ops:
            assert isinstance(op, Feature)
            assert op.frame_op

        self.frame_ops = frame_ops

    def set_seq(self, seq=None):
        """
        DESCRIPTION:
            update sequential operations list

        PARAMS:
            :seq: features to put in seq_ops
        """
        for op in seq_ops:
            assert isinstance(Feature)
        # Need to handle case of someone sending a batch into frame op
        # & vise versa
        self.seq_ops = seq_ops

    def set_empty_frame(self, batch_ops, frame_ops, seq_ops):
        """
        DESCRIPTION:
            creates an empty frame with the operation keys only

        PARAMS:
            :batch_ops: features to extract from batches
            :frame_ops: features to extract from frames
            :seq_ops: features to extract sequentially
        """
        assert isinstance(batch_ops, list)
        assert isinstance(frame_ops, list)
        assert isinstance(seq_ops, list)

        frame = {
            'input': {},
            'meta_data': {},
            'batch_features': {},
            'frame_features': {},
            'seq': {},
            'seq_features': {},
            'seq_output': {}}

        for op in batch_ops:
            frame['batch_features'].update({op.key_name: None})
        for op in frame_ops:
            frame['frame_features'].update({op.key_name: None})
        for op in seq_ops:
            frame['seq_features'].update({op.key_name: None})

        self.empty_frame = frame

    def extract(self, keep_input_data=True):
        """
        DESCRIPTION:
            extract all features

        PARAMS:
            :keep_input_data: boolean check whether we want to keep original
                             data
        """
        if self.batch_ops == self.frame_ops == self.seq_ops == []:
            raise ValueError('No features were specified for extraction.')
            self.extract_nonseq(self.data, self.batch_ops, self.frame_ops)

        self.set_empty_frame(self.batch_ops, self.frame_ops, self.seq_ops)
        self.output = []
        n_frame = 0
        n_batch = 0

        for batch in self.data:
            batch_dict = {}
            for op in self.batch_ops:
                batch_dict.update({op.key_name: op.extract(batch)})

            for frame in batch:
                frame_dict = copy.deepcopy(self.empty_frame)
                frame_dict['input'] = frame
                for op in self.frame_ops:
                    frame_dict['frame_features'][op.key_name] = op.extract(
                        frame)
                frame_dict['batch_features'].update(batch_dict)
                frame_dict['meta_data'].update({'frame_number': n_frame,
                                               'batch_number': n_batch})
                self.output.append(frame_dict)
                n_frame += 1
            n_batch += 1

        if self.seq_ops != [] and self.seq_ops is not None:
            for frame in self.output:
                temp = frame['input']
                for op in self.seq_ops:
                    temp = op.extract(temp)
                    if op.save is True or self.save_all is True:
                        frame['seq_features'].update({op.key_name: temp})
            frame['seq_output'] = temp

        if keep_input_data is False:
            self.data = []

        return self.output

    def as_ndarray(self, frame_key=None, batch_key=None, seq_key=None):
        """
        DESCRIPTION:
            get a feature as a numpy array

        PARAMS:
            :frame_key: key of frame feature to get
            :batch_key: key of batch feature to get
            :seq_key: key of seq operations to get
        """
        keys = [frame_key] + [batch_key] + [seq_key]

        key_count = 0
        for x in keys:
            if x is not None:
                key_count += 1
        if key_count != 1:
            raise ValueError('One and only one of the three keys may be set.')

        data = []
        if frame_key is not None:
            for frame_dict in self.output:
                data.append(frame_dict['frame_features'][frame_key])
        elif batch_key is not None:
            for frame_dict in self.output:
                data.append(frame_dict['batch_features'][batch_key])
        elif seq_key is not None:
            for frame_dict in self.output:
                data.append(frame_dict['seq_features'][seq_key])
        else:
            raise ValueError('Someone broke the as_ndarray method!')

        return np.array(data)

    def predict(self, X, model=''):
        """
        DESCRIPTION:
            predicts value for data X according to model. It returns a
            dictionary with the model name and predicted values as key and
            value

        PARAMS:
            :X: the data X to predict labels for
            :model: optional parameter to predict using a specific model only
                    or all if none is specified

        """
        assert len(self.models) > 0

        # we use dictionaries to return the predicted values with the
        # name of model that predicted it
        models_to_predict = {}
        predicted_values = {}

        if model == '':
            # if no model is specified just predict for all
            models_to_predict = self.models
        else:
            assert model in self.models
            # get specified model by getting it from the models dict
            models_to_predict.update({model: self.models[model]})

        for key, mod in models_to_predict.items():
            predicted_values.update({key: mod.predict(X)})

        return predicted_values

    def train(self, X, y, model=''):
        """
        DESCRIPTION:
            train the models using X as train data and y as labels for X

        PARAMS:
            :X: data to train on
            :y: labels for X
            :model: optional parameter to predict using a specific model only
                    or all if none is specified
        """
        assert len(self.models) > 0

        models_to_train = []

        if model == '':
            # if no model is specified just train all
            models_to_train = list(self.models.values())
        else:
            assert model in self.models
            # train specified model by getting it from the models dict
            models_to_train.append(self.models[model])

        for m in models_to_train:
            m.train(X, y)

    def display(self):
        """
        DESCRIPTION:
            print pipeline stuff like models we have and operations
        """
        print('models: {}'.format(self.models))
        print('save_all: {}'.format(self.save_all))
        print('batch_ops: {}'.format(self.batch_ops))
        print('frame_ops: {}'.format(self.frame_ops))
        print('seq_ops: {}'.format(self.seq_ops))
        print('empty frame: {}'.format(self.empty_frame))
        print('This method will not print output data.')
