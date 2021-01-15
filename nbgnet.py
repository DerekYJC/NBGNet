"""
NBGNet - NeuroBondGraph Network

NBGNet is a dynamical system incorporating both biological-inspired components 
and deep learning techniques to capture cross-scale dynamics that can infer 
the neural data from multiple scales.
"""

from __future__ import print_function
from keras import Model, activations
from keras.layers import Layer, Input, Concatenate, Dense
from keras.layers.recurrent import RNN
from keras import backend as K
import itertools
import numpy as np

class ForwardTransmission(Layer):
    """
    LFP - screwECoG forward transmission NBG-based RNN cell.
    """    

    def __init__(self, hidden_node=7, components = ['C1_inverse', 'C2_inverse', 'C3_inverse', 
                            'R1_inverse', 'R2_inverse', 'R3_inverse', 
                            'R5_inverse', 'R_Total_inverse', 'RICME'], **kwargs):
    
        """
        Create a forward transmission layer.
        
        Args:
            hidden_node: Positive integer, dimensionality of the hidden nodes.
            components: List, a list of modeled electrical components 
                (e.g. ['C1_inverse', 'C2_inverse', 'R1_inverse', ...])
        """
        self.hidden_node = hidden_node
        self.activation = activations.get('tanh')
        self.state_size = (1, 1, 1)
        self.fs = 3051.76   
        self.components = components
        super(ForwardTransmission, self).__init__(**kwargs)
        self.trainable = True
        
    def build(self, input_shape):
        
        # Create trainable weight variables 
        self.mapping_input_weight = {}
        self.mapping_input_bias = {}
        self.mapping_output_weight = {}
        self.mapping_output_bias = {}
        
        # Number of layers depends on the number of components 
        for component in self.components:
            self.mapping_input_weight[component] = self.add_weight(
                    name='mapping_input_weight_' + component,
                    shape=(input_shape[-1], self.hidden_node,),
                    initializer='uniform',
                    trainable=True)
            self.mapping_input_bias[component] = self.add_weight(
                    name='mapping_input_bias_' + component,
                    shape=(self.hidden_node,),
                    initializer='uniform',
                    trainable=True)
            self.mapping_output_weight[component] = self.add_weight(
                    name='mapping_output_weight_' + component,
                    shape=(self.hidden_node, 1, ),
                    initializer='uniform',
                    trainable=True)
            self.mapping_output_bias[component] = self.add_weight(
                    name='mapping_output_bias_' + component,
                    shape=(1,),
                    initializer='uniform',
                    trainable=True) 
            
        super(ForwardTransmission, self).build(input_shape)  
    
    def call(self, x, states):
        """
        LFP - screwECoG forward transmission function which implements the 
        electrophysiologically-inspired dynamical systems.
        
        Args:
            inputs: The input to the ForwardTransmission cell.
            state: The previous state from the last time step.
            
        Returns:
            An output and a list of state, where state is the newly computed 
            satae at time t.            
        """
    
        prev_q1 = states[0]
        prev_q2 = states[1]
        prev_q3 = states[2]
        
        # Nonlinear approximation for C1 -> V1
        h1_C1i = K.dot(prev_q1, self.mapping_input_weight['C1_inverse'])
        o1_C1i = h1_C1i + self.mapping_input_bias['C1_inverse']
        o1_C1i = self.activation(o1_C1i)
        h2_C1i = K.dot(o1_C1i, self.mapping_output_weight['C1_inverse'])
        o2_C1i = h2_C1i + self.mapping_output_bias['C1_inverse']
        V1 = self.activation(o2_C1i)
        
        # Nonlinear approximation for C2 -> V2
        h1_C2i = K.dot(prev_q2, self.mapping_input_weight['C2_inverse'])
        o1_C2i = h1_C2i + self.mapping_input_bias['C2_inverse']
        o1_C2i = self.activation(o1_C2i)
        h2_C2i = K.dot(o1_C2i, self.mapping_output_weight['C2_inverse'])
        o2_C2i = h2_C2i + self.mapping_output_bias['C2_inverse']
        V2 = self.activation(o2_C2i)

        # Nonlinear approximation for C3 -> V3
        h1_C3i = K.dot(prev_q3, self.mapping_input_weight['C3_inverse'])
        o1_C3i = h1_C3i + self.mapping_input_bias['C3_inverse']
        o1_C3i = self.activation(o1_C3i)
        h2_C3i = K.dot(o1_C3i, self.mapping_output_weight['C3_inverse'])
        o2_C3i = h2_C3i + self.mapping_output_bias['C3_inverse']
        V3 = self.activation(o2_C3i)
        
        # Nonlinear approximation for V1 -> I1
        h1_R1i = K.dot(V1, self.mapping_input_weight['R1_inverse'])
        o1_R1i = h1_R1i + self.mapping_input_bias['R1_inverse']
        o1_R1i = self.activation(o1_R1i)
        h2_R1i = K.dot(o1_R1i, self.mapping_output_weight['R1_inverse'])
        o2_R1i = h2_R1i + self.mapping_output_bias['R1_inverse']
        I1 = self.activation(o2_R1i)
        
        # Nonlinear approximation for V2 -> I2
        h1_R2i = K.dot(V2, self.mapping_input_weight['R2_inverse'])
        o1_R2i = h1_R2i + self.mapping_input_bias['R2_inverse']
        o1_R2i = self.activation(o1_R2i)
        h2_R2i = K.dot(o1_R2i, self.mapping_output_weight['R2_inverse'])
        o2_R2i = h2_R2i + self.mapping_output_bias['R2_inverse']
        I2 = self.activation(o2_R2i)
        
        # Nonlinear approximation for V3 -> I3
        h1_R3i = K.dot(V3, self.mapping_input_weight['R3_inverse'])
        o1_R3i = h1_R3i + self.mapping_input_bias['R3_inverse']
        o1_R3i = self.activation(o1_R3i)
        h2_R3i = K.dot(o1_R3i, self.mapping_output_weight['R3_inverse'])
        o2_R3i = h2_R3i + self.mapping_output_bias['R3_inverse']
        I3 = self.activation(o2_R3i)

        # Nonlinear approximation for (V2 - V3) -> I5
        h1_R5i = K.dot(V2 - V3, self.mapping_input_weight['R5_inverse'])
        o1_R5i = h1_R5i + self.mapping_input_bias['R5_inverse']
        o1_R5i = self.activation(o1_R5i)
        h2_R5i = K.dot(o1_R5i, self.mapping_output_weight['R5_inverse'])
        o2_R5i = h2_R5i + self.mapping_output_bias['R5_inverse']
        I5 = self.activation(o2_R5i)
        
        # Nonlinear approximation for (V_LFP - V1 - V2) -> IT
        h1_RTi = K.dot(x - V1 - V2, self.mapping_input_weight['R_Total_inverse'])
        o1_RTi = h1_RTi + self.mapping_input_bias['R_Total_inverse']
        o1_RTi = self.activation(o1_RTi)
        h2_RTi = K.dot(o1_RTi, self.mapping_output_weight['R_Total_inverse'])
        o2_RTi = h2_RTi + self.mapping_output_bias['R_Total_inverse']
        IT = self.activation(o2_RTi)
        
        # Obtain the state dot
        q1dot = IT - I1 
        q2dot = IT - I2 - I5
        q3dot = I5 - I3
        
        # Update the states 
        q1 = prev_q1 + q1dot / self.fs
        q2 = prev_q2 + q2dot / self.fs
        q3 = prev_q3 + q3dot / self.fs
        
        # Nonlinear approximation for IT -> V_ICME
        h1_RICME = K.dot(IT, self.mapping_input_weight['RICME'])
        o1_RICME = h1_RICME + self.mapping_input_bias['RICME']
        o1_RICME = self.activation(o1_RICME)
        h2_RICME = K.dot(o1_RICME, self.mapping_output_weight['RICME'])
        o2_RICME = h2_RICME + self.mapping_output_bias['RICME']
        output = o2_RICME

        return output, [q1, q2, q3]

    def get_config(self):
        config = {'hidden_node': self.hidden_node,
                  'components': self.components}
        base_config = super(ForwardTransmission, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    

class InverseTransmission(Layer):
    """
    LFP - screwECoG inverse transmission NBG-based RNN cell.
    """    

    def __init__(self, hidden_node=7,components = ['C1_inverse', 'C2_inverse', 'C3_inverse', 
                            'R1_inverse', 'R2_inverse', 'R3_inverse', 
                            'R5_inverse', 'R_Total', 'RICME_inverse'], **kwargs):
        """
        Create an inverse transmission layer.
        
        Args:
            hidden_node: Positive integer, dimensionality of the hidden nodes.
            components: List, a list of modeled electrical components 
                (e.g. ['C1_inverse', 'C2_inverse', 'R1_inverse', ...])
        """
        self.hidden_node = hidden_node
        self.activation = activations.get('tanh')
        self.state_size = (1, 1, 1)
        self.fs = 3051.76   
        self.components = components
        super(InverseTransmission, self).__init__(**kwargs)
        self.trainable = True
        
    def build(self, input_shape):
       
        # Create trainable weight variables     
        self.mapping_input_weight = {}
        self.mapping_input_bias = {}
        self.mapping_output_weight = {}
        self.mapping_output_bias = {}
        
        # Number of layers depends on the number of components 
        for component in self.components:
            self.mapping_input_weight[component] = self.add_weight(
                    name='mapping_input_weight_' + component,
                    shape=(input_shape[-1], self.hidden_node,),
                    initializer='uniform',
                    trainable=True)
            self.mapping_input_bias[component] = self.add_weight(
                    name='mapping_input_bias_' + component,
                    shape=(self.hidden_node,),
                    initializer='uniform',
                    trainable=True)
            self.mapping_output_weight[component] = self.add_weight(
                    name='mapping_output_weight_' + component,
                    shape=(self.hidden_node, 1, ),
                    initializer='uniform',
                    trainable=True)
            self.mapping_output_bias[component] = self.add_weight(
                    name='mapping_output_bias_' + component,
                    shape=(1,),
                    initializer='uniform',
                    trainable=True) 
            
        super(InverseTransmission, self).build(input_shape)  
    
    def call(self, x, states):
        """
        LFP - screwECoG inverse transmission function which implements the 
        electrophysiologically-inspired dynamical systems.
        
        Args:
            inputs: The input to the InverseTransmission cell.
            state: The previous state from the last time step.
            
        Returns:
            An output and a list of state, where state is the newly computed 
            satae at time t.            
        """
    
        prev_q1 = states[0]
        prev_q2 = states[1]
        prev_q3 = states[2]
        
        # Nonlinear approximation for C1 -> V1
        h1_C1i = K.dot(prev_q1, self.mapping_input_weight['C1_inverse'])
        o1_C1i = h1_C1i + self.mapping_input_bias['C1_inverse']
        o1_C1i = self.activation(o1_C1i)
        h2_C1i = K.dot(o1_C1i, self.mapping_output_weight['C1_inverse'])
        o2_C1i = h2_C1i + self.mapping_output_bias['C1_inverse']
        V1 = self.activation(o2_C1i)
        
        # Nonlinear approximation for C2 -> V2
        h1_C2i = K.dot(prev_q2, self.mapping_input_weight['C2_inverse'])
        o1_C2i = h1_C2i + self.mapping_input_bias['C2_inverse']
        o1_C2i = self.activation(o1_C2i)
        h2_C2i = K.dot(o1_C2i, self.mapping_output_weight['C2_inverse'])
        o2_C2i = h2_C2i + self.mapping_output_bias['C2_inverse']
        V2 = self.activation(o2_C2i)

        # Nonlinear approximation for C3 -> V3
        h1_C3i = K.dot(prev_q3, self.mapping_input_weight['C3_inverse'])
        o1_C3i = h1_C3i + self.mapping_input_bias['C3_inverse']
        o1_C3i = self.activation(o1_C3i)
        h2_C3i = K.dot(o1_C3i, self.mapping_output_weight['C3_inverse'])
        o2_C3i = h2_C3i + self.mapping_output_bias['C3_inverse']
        V3 = self.activation(o2_C3i)
        
        # Nonlinear approximation for V1 -> I1
        h1_R1i = K.dot(V1, self.mapping_input_weight['R1_inverse'])
        o1_R1i = h1_R1i + self.mapping_input_bias['R1_inverse']
        o1_R1i = self.activation(o1_R1i)
        h2_R1i = K.dot(o1_R1i, self.mapping_output_weight['R1_inverse'])
        o2_R1i = h2_R1i + self.mapping_output_bias['R1_inverse']
        I1 = self.activation(o2_R1i)
        
        # Nonlinear approximation for V2 -> I2
        h1_R2i = K.dot(V2, self.mapping_input_weight['R2_inverse'])
        o1_R2i = h1_R2i + self.mapping_input_bias['R2_inverse']
        o1_R2i = self.activation(o1_R2i)
        h2_R2i = K.dot(o1_R2i, self.mapping_output_weight['R2_inverse'])
        o2_R2i = h2_R2i + self.mapping_output_bias['R2_inverse']
        I2 = self.activation(o2_R2i)
        
        # Nonlinear approximation for V3 -> I3
        h1_R3i = K.dot(V3, self.mapping_input_weight['R3_inverse'])
        o1_R3i = h1_R3i + self.mapping_input_bias['R3_inverse']
        o1_R3i = self.activation(o1_R3i)
        h2_R3i = K.dot(o1_R3i, self.mapping_output_weight['R3_inverse'])
        o2_R3i = h2_R3i + self.mapping_output_bias['R3_inverse']
        I3 = self.activation(o2_R3i)

        # Nonlinear approximation for (V2 - V3) -> I5
        h1_R5i = K.dot(V2 - V3, self.mapping_input_weight['R5_inverse'])
        o1_R5i = h1_R5i + self.mapping_input_bias['R5_inverse']
        o1_R5i = self.activation(o1_R5i)
        h2_R5i = K.dot(o1_R5i, self.mapping_output_weight['R5_inverse'])
        o2_R5i = h2_R5i + self.mapping_output_bias['R5_inverse']
        I5 = self.activation(o2_R5i)
        
        # Nonlinear approximation for V_ICME -> I_ICME
        h1_RICMEi = K.dot(x, self.mapping_input_weight['RICME_inverse'])
        o1_RICMEi = h1_RICMEi + self.mapping_input_bias['RICME_inverse']
        o1_RICMEi = self.activation(o1_RICMEi)
        h2_RICMEi = K.dot(o1_RICMEi, self.mapping_output_weight['RICME_inverse'])
        o2_RICMEi = h2_RICMEi + self.mapping_output_bias['RICME_inverse']
        IICME = self.activation(o2_RICMEi)
        
        # Obtain the state dot
        q1dot = IICME - I1 
        q2dot = IICME - I2 - I5
        q3dot = I5 - I3
        
        # Update the states 
        q1 = prev_q1 + q1dot / self.fs
        q2 = prev_q2 + q2dot / self.fs
        q3 = prev_q3 + q3dot / self.fs
        
        # Nonlinear approximation for I_ICME -> V_T
        h1_RT = K.dot(IICME, self.mapping_input_weight['R_Total'])
        o1_RT = h1_RT + self.mapping_input_bias['R_Total']
        o1_RT = self.activation(o1_RT)
        h2_RT = K.dot(o1_RT, self.mapping_output_weight['R_Total'])
        o2_RT = h2_RT + self.mapping_output_bias['R_Total']
        VT = self.activation(o2_RT)
        
        output = V1 + V2 + VT

        return output, [q1, q2, q3]

    def get_config(self):
        config = {'hidden_node': self.hidden_node,
                  'components': self.components}
        base_config = super(InverseTransmission, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def build_model(mode1_channels=157, mode2_channels=16, loadWeight=False, 
                forwardWeightsFileName='', inverseWeightsFileName='',
                ):
    """
    Function to build the NBGNet model.
    Args:
        mode1_channels: Positive integer, number of mode 1 (e.g., LFP) channels
        mode2_channels: Positive integer, number of mode 2 (e.g., screwECoG) channels
        loadWeight: Binary, whether load the pre-trained weight or not
        forwardWeightsFileName: String, path of the saved .h5 file for forward NBGNet's weights    
        inverseWeightsFileName: String, path of the saved .h5 file for inverse NBGNet's weights    
        
    Returns:
        Forward_NBGNet: NBGNet model for forward modeling.         
        Inverse_NBGNet: NBGNet model for inverse modeling.             
    """
    
    # n_models = mode1_channels * mode2_channels
    # print("Total " + str(n_models) + " models to be built!!!")   
    # ----- Forward NBGNet -------------------------------------
    # Initialize the dictionary and array to put 
    ForwardModels, ForwardModel_RNNs = {}, {}
    y, q1, q2, q3 = {}, {}, {}, {}
    x, outputs = [], []
    
    # Organize the input list
    for inputs in range(mode1_channels):
        x.append(Input(shape=(None, 1)))        
    
    # Call the models with the corresponding input channel and output channel
    for input_ch, output_ch in itertools.product(range(1, mode1_channels+1), 
                                                 range(1, mode2_channels+1)):
        ForwardModels[(input_ch, output_ch)] = ForwardTransmission(name='LFP_' + str(input_ch) + '--ICME_' + str(output_ch))   
        ForwardModel_RNNs[(input_ch, output_ch)] = RNN(ForwardModels[(input_ch, output_ch)], 
                         return_state=True)
        input_model = x[input_ch - 1]
        y[(input_ch, output_ch)], q1[(input_ch, output_ch)], q2[(input_ch, output_ch)], q3[(input_ch, output_ch)] = ForwardModel_RNNs[(input_ch, output_ch)](input_model)
        # if output_ch == mode2_channels:
        #     print("Input channel #" + str(input_ch) + " has connected to all the output!!")
        
    # Obtain the output list
    for output in range(mode2_channels):
        Y = y[(1, output+1)]
        for inputs in range(1, mode1_channels):
            Y = Concatenate()([Y, y[(inputs+1, output+1)]])
        outputs.append(Dense(1, use_bias=False)(Y))
    
    # Compile the model and display the summary of the model
    Forward_NBGNet = Model(inputs=x, outputs=outputs)    
    Forward_NBGNet.compile(optimizer='adam', loss='mse')

    # ----- Inverse NBGNet -------------------------------------
    # Initialize the dictionary and array to put 
    InverseModels, InverseModel_RNNs = {}, {}
    y_, q1_, q2_, q3_ = {}, {}, {}, {}
    x_, outputs_ = [], []
    
    # Organize the input list
    for inputs in range(mode2_channels):
        x_.append(Input(shape=(None, 1)))
        
    # Call the models with the corresponding input channel and output channel
    for input_ch, output_ch in itertools.product(range(1, mode2_channels+1), 
                                                 range(1, mode1_channels+1)):
        InverseModels[(input_ch, output_ch)] = InverseTransmission(name='ICME_' + str(input_ch) + '--LFP_' + str(output_ch))   
        InverseModel_RNNs[(input_ch, output_ch)] = RNN(InverseModels[(input_ch, output_ch)], 
                         return_state=True)
        input_model = x_[input_ch - 1]
        y_[(input_ch, output_ch)], q1_[(input_ch, output_ch)], q2_[(input_ch, output_ch)], q3_[(input_ch, output_ch)] = InverseModel_RNNs[(input_ch, output_ch)](input_model)
        # if output_ch == output_channels:
        #     print("Input channel #" + str(input_ch) + " has connected to all the output!!")
    
    # Obtain the output list
    for output in range(mode1_channels):
        Y = y_[(1, output+1)]
        for inputs in range(1, mode2_channels):
            Y = Concatenate()([Y, y_[(inputs+1, output+1)]])
        outputs_.append(Dense(1, use_bias=False)(Y))
    
    # Compile the model and display the summary of the model
    Inverse_NBGNet = Model(inputs=x_, outputs=outputs_)    
    Inverse_NBGNet.compile(optimizer='adam', loss='mse')
        
    if loadWeight:
        Forward_NBGNet.load_weights(forwardWeightsFileName)
        Inverse_NBGNet.load_weights(inverseWeightsFileName)
    
    return Forward_NBGNet, Inverse_NBGNet

def train_model(model, X_train, y_train, epochsPerTrial, nRounds, batchSize,
                saveResults=True, 
                saveFileName=''):
    """
    Function to train the NBGNet model.
    Args:
        model: model to be trained
        X_train: Dictionary, where single-trial input data is stored in corresponding trial as key
        y_train: Dictionary, where single-trial output data is stored in corresponding trial as key
        epochsPerTrial: Integer, number of epochs per trial for model training    
        nRounds: Integer, number of rounds for model training     
        batchSize: Integer, batch size for model training  
        saveResults: Binary, whether save the trained weight or not    
        saveFileName: String, path of the saved .h5 file      
        
    Returns:
        history: training history         
    """
    
    history = {}
    for r in range(nRounds):
        for tr in range(len(X_train.keys())):
            history[(r, tr)] = model.fit(X_train[tr], y_train[tr], batch_size=batchSize, 
                                         epochs=epochsPerTrial)
    
    if saveResults:
        model.save_weights(saveFileName)
    
    return history

def evaluate_model(model, X_test, y_test, evaluationfunc,
                saveResults=True, saveFileName=''):
    """
    Function to train the NBGNet model.
    Args:
        model: model to be trained
        X_train: Dictionary, where single-trial input data is stored in corresponding trial as key
        y_train: Dictionary, where single-trial output data is stored in corresponding trial as key
        evaluationfunc: Function or lists of functions to evluate the model performance    
        saveResults: Binary, whether save the results based on evaluation function or not    
        saveFileName: String, path of the saved       
        
    Returns:
        evaluation_results: model evaluation with specified metrics       
    """
    
    y_pred = model.predict(X_test) 
    
    if callable(evaluationfunc):
        evaluation_results = evaluationfunc(y_pred, y_test)
    elif type(evaluationfunc) is list: 
        evaluation_results = dict()
        for i, func in enumerate(evaluationfunc):
            evaluation_results[i] = func(y_pred, y_test)
    
    if saveResults:
        np.save(saveFileName, evaluation_results)
    
    return evaluation_results

# Example evaluation metrics: mse, cc
def mean_squared_error_func(y_pred, y_test):
    mse = np.mean(np.square(y_pred - y_test))
    return mse

def cross_correlation_func(y_pred, y_test):
    cc = np.corrcoef(y_pred, y_test)[0, 1]
    return cc
