# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 17:23:21 2018

@author: RAGHAV
"""
import numpy as np
import matplotlib.pyplot as plt
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}

plt.rcParams.update(params)
import tensorflow as tf

def make_prediction(model=None,img_vector=[],
                    label_dict={},top_N=3,
                    model_input_shape=None):
    if model:
        prediction = model.predict(img_vector)
        prediction = tf.nn.sigmoid(prediction)

        kind = np.sort(prediction[0])[::-1][:top_N]

        confidence_predicted = kind if kind >= 0.5 else 1 - kind

        prediction = tf.where(prediction < 0.5, 0, 1)
        prediction = prediction.numpy()

        labels_predicted = label_dict.get(prediction[0][0])


        return labels_predicted, confidence_predicted


def plot_predictions(model,dataset,
                    dataset_labels,label_dict,
                    batch_size,grid_height,grid_width):
    if model:
        f, ax = plt.subplots(grid_width, grid_height)
        f.set_size_inches(12, 12)

        random_batch_indx = np.random.permutation(np.arange(0,len(dataset)))[:batch_size]

        img_idx = 0
        for i in range(0, grid_width):
            for j in range(0, grid_height):
                actual_label = label_dict.get(dataset_labels[random_batch_indx[img_idx]])
                preds, confs_ = make_prediction(model,
                                              img_vector=dataset[random_batch_indx[img_idx]:random_batch_indx[img_idx]+1],
                                              label_dict=label_dict,
                                              top_N=1)
                ax[i][j].axis('off')
                ax[i][j].set_title('Actual:'+actual_label[:10]+\
                                    '\nPredicted:'+preds + \
                                    '(' +str(round(confs_[0],2)) + ')', color='black')
                ax[i][j].imshow(dataset[random_batch_indx[img_idx]])
                img_idx += 1

        plt.subplots_adjust(left=0, bottom=0, right=1,
                            top=1, wspace=0.4, hspace=0.55)
 

# source: https://github.com/keras-team/keras/issues/431#issuecomment-317397154
def get_activations(model, model_inputs, 
    print_shape_only=True, layer_name=None):
    import tensorflow.keras.backend as K
    print('----- activations -----')
    activations = []
    inp = model.input

    #调整维度
    model_inputs = np.expand_dims(model_inputs,axis=0)
    
    model_multi_inputs_cond = True
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False
    # all layer outputs
    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]  

    # evaluation functions           
    #funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs] 
    funcs = [K.function(inp, [out]) for out in outputs] 

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(1.)
    else:
        list_inputs = [model_inputs, 1.]

    # Learning phase. 1 = Test mode (no dropout or batch normalization)
    layer_outputs = [func([model_inputs])[0] for func in funcs]
    #layer_outputs = [func(list_inputs)[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations

# source :https://github.com/philipperemy/keras-visualize-activations/blob/master/read_activations.py
def display_activations(activation_maps):
    batch_size = activation_maps[0].shape[0]
    assert batch_size == 1, 'One image at a time to visualize.'
    for i, activation_map in enumerate(activation_maps):
        print('Displaying activation map {}'.format(i))
        shape = activation_map.shape
        if len(shape) == 4:
            activations = np.hstack(np.transpose(activation_map[0], (2, 0, 1)))
        elif len(shape) == 2:
            # try to make it square as much as possible. we can skip some activations.
            activations = activation_map[0]
            num_activations = len(activations)
            # too hard to display it on the screen.
            if num_activations > 1024:  
                square_param = int(np.floor(np.sqrt(num_activations)))
                activations = activations[0: square_param * square_param]
                activations = np.reshape(activations, (square_param, square_param))
            else:
                activations = np.expand_dims(activations, axis=0)
        else:
            raise Exception('len(shape) = 3 has not been implemented.')
        #plt.imshow(activations, interpolation='None', cmap='binary')
        fig, ax = plt.subplots(figsize=(18, 12))
        ax.imshow(activations, interpolation='None', cmap='binary')
        plt.show()        
