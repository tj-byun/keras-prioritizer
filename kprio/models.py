# first: 11/05/18
# last: 12/8/18
# Taejoon Byun

from __future__ import print_function
import keras
import numpy as np
import logging

logger = logging.getLogger('kprio')

def load_model(kind, path, **kwargs):
    """ A proxy to keras.models.load_model. """
    supported_model_dict = {"taxinet": TaxiNet, "taxinet1": TaxiNet1, "taxinet5": TaxiNet5}
    new_kwargs = kwargs
    if "taxinet" in kind.lower():
        from taxinet_utils.loss import smoothL1
        new_kwargs = dict(kwargs, custom_objects={'smoothL1':smoothL1})
    m = keras.models.load_model(path, **new_kwargs)
    m.summary()
    if kind in supported_model_dict:
        m.__class__ = supported_model_dict[kind]
    else:
        m.__class__ = Model
    return m



class Model(keras.models.Model):

    def __init__(self, *args, **kwargs):
        super(Model, *args, **kwargs).__init__()


    def __get_trainable_layers(self):
        return filter(lambda l: len(l.trainable_weights) > 0, self.layers)


    def get_feature_layer(self, offset=0):
        """ Returns the 'flatten layer' (the output of the last
        flatten layer can be interpreted as a feature vector) """
        last_conv = -1
        fisrt_dense = -1
        for i, layer in enumerate(self.layers):
            if isinstance(layer, keras.layers.convolutional._Conv):
                last_conv = i
            elif 'Pool' in str(type(layer)):
                last_conv = i
            elif isinstance(layer, keras.layers.Dense):
                first_dense = i
                break
        if last_conv == -1:
            raise Exception("Not a convolutional neural network!")
        logger.debug("feature layer: " + 
                str(self.layers[first_dense + offset - 1]))
        return self.layers[first_dense + offset - 1]


    def predict_and_process(self, *args, **kwargs):
        return self.predict(*args, **kwargs)


    #TODO (12/10): implement a more flexible layer selection.
    def get_activation_traces(self, input_data, selection, bucket_size=None):
        """ Obtain activation traces from the selected layers for the given
        input date.
        :param input_data: Input data to run
        :param selection: Layers to select. A slice or one of ['feature', 'all']
        :param bucket_size: size of the input bucket to run at the same time
            (to prevent Keras from crashing with OOM error)
        """
        assert type(input_data) == np.ndarray
        ilen = input_data.shape[0]
        assert ilen > 0
        logger.info('Running {} input data'.format(ilen))

        layers = None
        if selection == 'feature':
            layers = [self.get_feature_layer()]
        elif selection == 'all':
            layers = self.__get_trainable_layers()
        elif type(selection) == slice:
            layers = self.__get_trainable_layers()[selection]
        else:
            raise Exception("Unknown selection criterion {}".format(selection))
        neuron_outs = [layer.output for layer in layers]

        # When the length of the `input_data` is larger than given
        # `bucket_size`, run the K.function multiple times in order to prevent
        # the GPU memory from overflowing.
        ats = None
        def get_flat_shape(arr):
            return reduce(lambda x, y: x * y, arr.shape[1:])
        for i in range(len(input_data) / bucket_size + 1):
            ind_a = i * bucket_size
            ind_b = min(len(input_data), (i + 1) * bucket_size)
            input_bucket = input_data[ind_a:ind_b]
            logger.debug('Bucket {} {}'.format(ind_a, ind_b))
            # at: activation trace per layer. List of `np.array`s
            at = keras.backend.function([self.input], neuron_outs)\
                    ([input_bucket])
            for i in range(len(at)):
                at[i] = np.array(at[i]).reshape(at[i].shape[0],
                        get_flat_shape(at[i]))
            if ats is None:
                ats = at
            else:
                for i in range(len(at)):
                    ats[i] = np.concatenate((ats[i], at[i]), axis=0)

        # transpose to [input x flat_actiation_trace]
        return np.concatenate((ats), axis=1)



class TaxiNet(Model):

    def predict_and_process(self, input_data, **kwargs):
        # The normalization of the neural net output is based upon a maximum
        # runway width of 32 meters (+/-16 from center) for the cross track
        # error, and +/- 30 degrees of heading error, the neural network
        # outputs normalized (0 to 1) values for both CTE and heading,
        # therefore ensure your scaling by 16 and 30 respectively when using
        # the neural net output
        scale = np.array([16.0, 32.0])
        outs = [self.predict(np.expand_dims(x, axis=0), **kwargs)
                for x in input_data]
        return np.squeeze(np.array(outs))


class TaxiNet5(Model):

    def predict_and_process(self, input_data, **kwargs):
        #scale = np.array([8.0, 35.0])
        outs = [self.predict(np.expand_dims(x/255.0, axis=0), **kwargs)
                for x in input_data]
        return np.squeeze(np.array(outs))


class TaxiNet1(Model):

    def predict_and_process(self, input_data, **kwargs):
        # The normalization of the neural net output is based upon a maximum
        # runway width of 32 meters (+/-16 from center) for the cross track
        # error, and +/- 30 degrees of heading error, the neural network
        # outputs normalized (0 to 1) values for both CTE and heading,
        # therefore ensure your scaling by 16 and 30 respectively when using
        # the neural net output
        outs = [self.predict(np.expand_dims(x/255.0, axis=0), **kwargs) * 16.0
                for x in input_data]
        return np.squeeze(np.array(outs))
