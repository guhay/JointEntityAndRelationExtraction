import tensorflow as tf

class JointExtraction():
    def __init__(self,batch_size,vocab_size,embed_size,rnn_size,sequence_max_length,num_class,learning_rate,optimizer,filter_size,num_filters,RC_num_class,RC_learning_rate,mode='RC'):
        self.batch_size=batch_size
        self.vocab_size=vocab_size
        self.embed_size=embed_size
        self.rnn_size=rnn_size
        self.sequence_max_length=sequence_max_length
        self.num_class=num_class
        self.learning_rate=learning_rate
        self.optimizer=optimizer
        self.filter_size=filter_size
        self.num_filters=num_filters
        self.RC_num_class=RC_num_class
        self.RC_learning_rate=RC_learning_rate
        self.mode=mode
        with tf.name_scope('NER_placeholder'):
            self.inputs=tf.placeholder(tf.int32,[None,self.sequence_max_length])
            self.sequence_length=tf.placeholder(tf.int32,[None])
            self.dropout_keep_prob=tf.placeholder(tf.float32,name='dropout_keep_prob')
            self.NER_target=tf.placeholder(tf.int32,shape=[None,self.sequence_max_length])
        self.init_weights()
        self.encoder_layer()
        self.decoder_NER()
        self.train_NER()
        self.RC_init()

    def init_weights(self):
        # self.embeding=tf.get_variable('Embeding',shape=[self.vocab_size,self.embed_size])
        self.embeding=tf.Variable(tf.truncated_normal([self.vocab_size,self.embed_size],stddev=0.1),name='embeding')

    def encoder_layer(self):
        with tf.name_scope('bilstm'):
            self.embeding_words = tf.nn.embedding_lookup(self.embeding,self.inputs)  # [batch_size,sequence_length,embeding_size]
            lstm_fw=tf.contrib.rnn.BasicLSTMCell(self.rnn_size)
            lstm_bw=tf.contrib.rnn.BasicLSTMCell(self.rnn_size)
            if self.dropout_keep_prob is not None:
                lstm_fw=tf.contrib.rnn.DropoutWrapper(lstm_fw,output_keep_prob=self.dropout_keep_prob)
                lstm_bw=tf.contrib.rnn.DropoutWrapper(lstm_bw,output_keep_prob=self.dropout_keep_prob)
            outputs,last_state=tf.nn.bidirectional_dynamic_rnn(lstm_fw,lstm_bw,self.embeding_words,sequence_length=self.sequence_length,dtype=tf.float32)
            output_bilstm=tf.concat(outputs,axis=2)
            self.output_bilstm=output_bilstm #[batch_size,max_sequence_length,rnn_size*2]
            if self.mode=='RC':
                self.embeding_H=tf.reduce_mean(self.output_bilstm,axis=0)

    def decoder_NER(self):
        with tf.name_scope('NER_decoder'):
            W_NER=tf.Variable(tf.random_normal(shape=[self.rnn_size,self.num_class], mean=0, stddev=1), name='W_NER')
            b_NER=tf.Variable(tf.constant(0.1, shape=[self.num_class]), name="b_NER")
            NER_cell=tf.contrib.rnn.BasicLSTMCell(self.rnn_size)
            NER_outputs,NER_laststate=tf.nn.dynamic_rnn(cell=NER_cell,inputs=self.output_bilstm,sequence_length=self.sequence_length,time_major=False,dtype=tf.float32)#[batch_size,max_sequence_length,rnn_size]
            # NER_outputs=tf.transpose(NER_outputs,perm=[1,0,2]) #[max_sequence_length,batch_size,rnn_size]
            with tf.name_scope('NER_DNN'):
                NER_outputs=tf.reshape(NER_outputs,[-1,self.rnn_size]) #[batch_size*sequence_length,rnn_size]
                output=tf.matmul(NER_outputs,W_NER)+b_NER #[batch_size*sequence_length,num_class]
                output=tf.reshape(output,[-1,self.sequence_max_length,self.num_class])
                self.NER_output=tf.argmax(output,axis=2) #[batch_size,sequence_length]
        with tf.name_scope('NER_loss'):
            mask=tf.sequence_mask(self.sequence_length,self.sequence_max_length,dtype=tf.float32,name='mask')
            self.batch_loss=tf.contrib.seq2seq.sequence_loss(
                logits=output,
                targets=self.NER_target,
                weights=mask,
                name='batch_loss'
            )
    def train_NER(self):
        with tf.name_scope('NER_train'):
            self.training_op=self.optimizer(self.learning_rate,name='training_op').minimize(self.batch_loss)

    def RC_init(self):
         with tf.name_scope('RC_placeholder'):
             self.begin2end = tf.placeholder(tf.int32, [self.sequence_max_length])
             self.entity1 = tf.placeholder(tf.int32, [None])
             self.entity2 = tf.placeholder(tf.int32, [None])
             self.targetRC = tf.placeholder(tf.int32, [None])
         with tf.name_scope('RC_input'):
             RCInputW=tf.nn.embedding_lookup(self.embeding, self.begin2end)
             RCE1=tf.reshape(tf.reduce_sum(tf.nn.embedding_lookup(self.embeding_H,self.entity1),axis=0),[1,-1])
             RCE2=tf.reshape(tf.reduce_sum(tf.nn.embedding_lookup(self.embeding_H,self.entity2),axis=0),[1,-1])
             RC_Embeding=tf.concat([RCE1,RCInputW,RCE2],0)
             RC_Embeding=tf.expand_dims(RC_Embeding,0)
             RC_Embeding=tf.expand_dims(RC_Embeding,-1)
             self.RC_Embeding=RC_Embeding #[1,sequence_length,embeding_size,1]
         with tf.name_scope('conv-maxpool'):
            filter_shape=[self.filter_size,self.embed_size,1,self.num_filters]
            W=tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name='W')
            b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
            conv=tf.nn.conv2d(
                self.RC_Embeding,
                W,
                strides=[1,1,1,1],
                padding='VALID',
                name='conv'
            )
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            pooled=tf.nn.max_pool(
                h,
                ksize=[1,self.sequence_max_length+2-self.filter_size+1,1,1],
                strides=[1,1,1,1],
                padding='VALID',
                name='pool'
            )
            h_pool_flat=tf.reshape(pooled,[-1,self.num_filters])
            self.h_pool_flat=tf.reshape(tf.squeeze(h_pool_flat),[-1,self.num_filters])
         with tf.name_scope('RC_output'):
             W=tf.Variable(tf.random_normal(shape=[self.num_filters,self.RC_num_class], mean=0, stddev=1), name='W')
             b = tf.Variable(tf.constant(0.1, shape=[self.RC_num_class]), name="b")
             self.RC_score=tf.nn.xw_plus_b(self.h_pool_flat,W,b,name='RC-score')
             self.RC=tf.argmax(self.RC_score,1,name='RC-prediction')
         with tf.name_scope('RCloss'):
             loss=tf.nn.softmax_cross_entropy_with_logits(logits=tf.expand_dims(self.RC_score,0),labels=tf.expand_dims(self.targetRC,0))
             self.RC_loss=tf.reduce_mean(loss)
         with tf.name_scope('RCTrain'):
             self.RC_training_op=self.optimizer(self.RC_learning_rate,name='training_op').minimize(self.RC_loss)