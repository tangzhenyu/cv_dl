import tensorflow as tf
import numpy as np

class SiameseLSTM(object):
    def extract_axis(self, data, ind):
        batch_range = tf.range(tf.shape(data)[0])
        indices = tf.stack([batch_range, ind], axis = 1)
        res = tf.gather_nd(data, indices)
        return res

    def LSTMcell(self, n_hidden, reuse=True):
        if  reuse:
            return tf.contrib.rnn.BasicLSTMCell(n_hidden, reuse=reuse)
            #GRU
            #return tf.contrib.rnn.GRUCell(n_hidden, reuse=reuse)
            #BASIC LSTM with peepholes
            #return tf.nn.rnn_cell.LSTMCell(n_hidden, use_peepholes=True, reuse=reuse)
        else:
            return tf.contrib.rnn.BasicLSTMCell(n_hidden)
            #GRU
            #return tf.contrib.rnn.GRUCell(n_hidden)
            #BASIC LSTM with peepholes
            #return tf.nn.rnn_cell.LSTMCell(n_hidden, use_peepholes=True)

    def BiRNN(self, x, dropout, scope, embedding_size, sequence_length, video_lengths, num_lstm_layers, hidden_unit_dim, reuse, return_outputs):
        n_input=embedding_size
        n_steps=sequence_length
        #n_hidden layer_ number of features
        n_hidden=hidden_unit_dim
        #num-layers of lstm n_layers=2 => input(t)-> lstm(1)->lstm(2)->output(t)
        n_layers=num_lstm_layers

        # Prepare data shape to match `bidirectional_rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input) (?, seq_len, embedding_size)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
        # Permuting batch_size and n_steps
        #x = tf.transpose(x, [1, 0, 2])
        # Reshape to (n_steps*batch_size, n_input)
        #x = tf.reshape(x, [-1, n_input])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        #x = tf.split(x, n_steps, axis = 0)
        # Define lstm cells with tensorflow
        # Forward direction cell
        with tf.name_scope("fw"+scope),tf.variable_scope("fw"+scope):
            stacked_rnn_fw = []
            for _ in range(n_layers):
                fw_cell = self.LSTMcell(n_hidden, reuse=reuse)
                lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell,output_keep_prob=dropout)
                stacked_rnn_fw.append(lstm_fw_cell)
            lstm_fw_cell_m = tf.contrib.rnn.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)
        # Backward direction cell
        with tf.name_scope("bw"+scope),tf.variable_scope("bw"+scope):
            stacked_rnn_bw = []
            for _ in range(n_layers):
                bw_cell = self.LSTMcell(n_hidden, reuse=reuse)
                #bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
                lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell,output_keep_prob=dropout)
                stacked_rnn_bw.append(lstm_bw_cell)
            lstm_bw_cell_m = tf.contrib.rnn.MultiRNNCell(cells=stacked_rnn_bw, state_is_tuple=True)

        # Get lstm cell output
        with tf.name_scope("bw"+scope),tf.variable_scope("bw"+scope):
            #outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, x, dtype=tf.float32)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_m, lstm_bw_cell_m, inputs= x, dtype=tf.float32,sequence_length=video_lengths)
            #print(outputs, states)
            outputs = tf.concat(outputs, 2)
            #print(outputs)
            outputs = self.extract_axis(outputs, video_lengths-1)
        if return_outputs == 0:
            print(outputs)
            return outputs
        elif return_outputs == 1:
            state_fw, state_bw = states
            print(state_fw[0].h)
            return tf.concat([state_fw[0].h,state_bw[0].h], axis=1)
        elif return_outputs == 2:
            state_fw, state_bw = states
            print(state_fw[0].c)
            return tf.concat([state_fw[0].c,state_bw[0].c], axis=1)
        else:
            raise ValueError('requested value of return_outputs missing')

    def contrastive_loss(self, y,d,batch_size):
        tmp= y *tf.square(d)
        tmp2 = (1-y) *tf.square(tf.maximum((1 - d),0))
        return tf.reduce_sum(tmp +tmp2)/batch_size/2

    def fc(self, input_, in_channels, out_channels, name, relu):
      input_ = tf.reshape(input_ , [-1, in_channels])
      with tf.variable_scope(name) as scope:
        filt = tf.get_variable('weights', shape=[in_channels , out_channels], trainable=True)
        bias = tf.get_variable('biases',  shape=[out_channels], trainable=True)
      if relu:
        return tf.nn.relu(tf.nn.bias_add(tf.matmul(input_, filt), bias))
      else:
        return tf.nn.bias_add(tf.matmul(input_, filt), bias)



    def __init__(
      self, sequence_length, input_size, embedding_size, l2_reg_lambda, batch_size, num_lstm_layers, hidden_unit_dim, loss, projection, return_outputs):

      # Placeholders for input, output and dropout
      self.input_x1 = tf.placeholder(tf.float32, [None, input_size], name="input_x1")
      self.input_x2 = tf.placeholder(tf.float32, [None, input_size], name="input_x2")
      self.input_y = tf.placeholder(tf.float32, [None], name="input_y")
      self.input_y_classification = tf.placeholder(tf.float32, [None, 3], name="input_y_classification")
      self.video_lengths = tf.placeholder(tf.int32, [None], name="video_lengths")
      self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

      # Keeping track of l2 regularization loss (optional)
      l2_loss = tf.constant(0.0, name="l2_loss")

      # Add a projection_layer
      if projection:
        with tf.name_scope("projection"):
          self.projection_weights = tf.get_variable("projection", shape=[input_size, embedding_size], initializer=tf.contrib.layers.xavier_initializer())
          self.embedding1 = tf.matmul(self.input_x1, self.projection_weights)
          self.embedding2 = tf.matmul(self.input_x2, self.projection_weights)
      else:
        with tf.name_scope("projection"):
          embedding_size = input_size
          self.embedding1 = self.input_x1
          self.embedding2 = self.input_x2

      self.embedding1 = tf.reshape(self.embedding1, tf.convert_to_tensor([-1, sequence_length, embedding_size]))
      self.embedding2 = tf.reshape(self.embedding2, tf.convert_to_tensor([-1, sequence_length, embedding_size]))

      # Create a convolution + maxpool layer for each filter size
      with tf.name_scope("output"):
        self.out1=self.BiRNN(self.embedding1, self.dropout_keep_prob, "side1", embedding_size, sequence_length, self.video_lengths, num_lstm_layers=num_lstm_layers, hidden_unit_dim=hidden_unit_dim, reuse=False, return_outputs=return_outputs)
        self.out2=self.BiRNN(self.embedding2, self.dropout_keep_prob, "side1", embedding_size, sequence_length, self.video_lengths, num_lstm_layers=num_lstm_layers, hidden_unit_dim=hidden_unit_dim, reuse=True, return_outputs=return_outputs)


      # define distance and loss functions
      epsilon = tf.constant(1e-30)
      if loss == "AAAI":
        with tf.name_scope("output"):
          self.distance = tf.reduce_sum(tf.abs(tf.subtract(self.out1,self.out2)),1,keep_dims=True)
          self.distance = tf.reshape(self.distance, [-1])
          self.distance = tf.exp(-self.distance, name="distance")
        with tf.name_scope("loss"):
          self.loss = tf.losses.mean_squared_error(self.input_y, self.distance)/batch_size
      elif loss == "contrastive":
        #with tf.name_scope("output"):
        self.distance_feature = tf.add((tf.square(tf.subtract(self.out1,self.out2))), epsilon)
        self.distance = tf.sqrt(epsilon + tf.reduce_sum(tf.square(tf.subtract(self.out1,self.out2)),1,keep_dims=True))
        self.distance = tf.div(self.distance, tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.out1),1,keep_dims=True)),tf.sqrt(tf.reduce_sum(tf.square(self.out2),1,keep_dims=True))))
        self.distance = tf.reshape(self.distance, [-1], name="distance")
        print(self.distance)
        with tf.name_scope("loss"):
          self.distance_loss = self.contrastive_loss(self.input_y, self.distance, batch_size)
        #fc-layers and classification loss
        with tf.name_scope("fc_output"):
          self.fc1 = self.fc(self.distance_feature, hidden_unit_dim*2, 20, 'fc1', relu=True)
          self.fc2 = self.fc(self.fc1 , 20, 3, 'fc2', relu=False)
        with tf.name_scope("loss"):
          self.classification_loss = tf.softmax_cross_entropy_with_logits(labels=self.input_y_classification, logits=self.fc2, name='classification')
          self.loss = tf.add(self.distance_loss, self.classification_loss, name='siamese_total_loss')

      else:
        raise ValueError(" Loss function is not-defined")

      tf.summary.scalar('loss', self.loss)
