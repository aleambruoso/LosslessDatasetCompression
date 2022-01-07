import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
import pandas as pd
import six
import sys
sys.modules['sklearn.externals.six'] = six
import networkx as nx
import tensorflow as tf
from dahuffman import HuffmanCodec
import warnings
from tqdm import tqdm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
tf.compat.v1.disable_eager_execution()
tf.compat.v1.reset_default_graph()



#MODE= train or encode or decode
#TRAIN_PATH= if MODE == train or MODE == encode train file path ELSE encoded file
#TEST_PATH= if MODE == train or MODE == encode test file path ELSE nothing
def main(MODE='encode', TRAIN_PATH='./', TEST_PATH='./'):

    if(MODE=='encode' or MODE=='train'):

        if((TEST_PATH!='' and MODE=='encode')or MODE=='train'):

            #LOAD TEST AND TRAIN
            train_X= np.load(TRAIN_PATH)
            test_X= np.load(TEST_PATH)
        
        elif(MODE=='encode' and TEST_PATH==''):
            #LOAD DATASET
            train_X= np.load(TRAIN_PATH)

        #LOAD DATASET AND PREPROCESSING
        n_samples=train_X.shape[0]
        width= train_X.shape[1]
        height= train_X.shape[2]

        if(MODE=='encode' and TEST_PATH!=''):
            #JOIN TRAIN AND TEST
            n_samples+=test_X.shape[0]
            train_X=np.concatenate((train_X, test_X),0)


        train_X= train_X.reshape((n_samples, width*height)).astype('float32')

        #KNN-MST
        knn = kneighbors_graph(train_X, width, mode='distance', metric='euclidean')
        knn_mst= minimum_spanning_tree(knn) 

        #TSP
        g= nx.from_scipy_sparse_matrix(knn_mst)
        path= nx.dfs_preorder_nodes(g,0)
        tourOpt=list(path)

        #REORDER DATASET
        df= pd.DataFrame(train_X)
        newdf=df.loc[tourOpt]
        newTrain= newdf.to_numpy()


        if(MODE=='train'):
            #PREPARE DATASET FOR TRAINING
            test_X= test_X.reshape((test_X.shape[0], width*height))
            Training(newTrain, test_X)
        else:
            error_string, pixels= Predict(newTrain, width, height, MODE)

            print("end prediction")

            #HUFFMAN CODING
            errors=np.array(error_string)
            errors=errors.reshape(-1)
            length= errors.shape[0]
            codec=HuffmanCodec.from_data(errors)
            all_err= np.concatenate((errors, pixels),0)
            encoded=codec.encode(all_err)

            #meta informations
            meta=[n_samples, width, height, length, codec]     
            #save to file 0:meta(n_samples, width, height, errorLength) 1:encoded(errors, firstPixels)
            toSave= np.array([meta, encoded], dtype=object)
            np.save('datasetEncoded.npy', toSave)
    else:
        #LOAD AND DECODE DATASET
        dataset= np.load(TRAIN_PATH, allow_pickle=True )
        meta=dataset[0]
        encoded= dataset[1]
        codec=meta[4]
        all_dataset=codec.decode(data= encoded)
        errors= all_dataset[:meta[3]]
        pixels= all_dataset[meta[3]:]

        #PREPARE FOR PREDICTION
        errors= np.array(errors)
        pixels= np.array(pixels)
        errors= errors.reshape((meta[0], meta[1]*meta[2]))
        pixels= pixels.reshape((meta[0], 1))

        #PREDICT ORIGINAL DATASET
        originalImages, _= Predict(errors, meta[1], meta[2], MODE, pixels)
        newDataset= np.array(originalImages)
        newDataset= newDataset.reshape((meta[0], meta[1], meta[2]))

        #SAVE MODEL
        np.save('dataset.npy', newDataset)



def Training(train, test):
    #MODEL TRAINING
    samples= train.shape[0]
    dim= train.shape[1]
    tf.compat.v1.reset_default_graph()
    learning_rate  = 0.003							
    batch_size     = 250							
    number_epochs  = 80							
    number_input   = 1							
    number_steps   = dim					
    number_classes = 1						
    number_hidden  = 128							
    train_accuracy = test_accuracy = train_loss = test_loss = np.array([])

    # binarize the values of pixels to be either 0 or 1
    def binarize(images, threshold=0.1):
        return (threshold < images).astype('float32')


    def batch_by_batch_evaluetion(data_set):
        batch_accuracy = batch_loss = 0

        temp_samples= data_set.shape[0]

        # compute the number of total batches 
        total_batches = int(temp_samples / batch_size)

        # pass all the images batch by batch to compute the accuracy and loss
        prevBatch1=0
        actualBatch1=batch_size
        for _ in range(total_batches):
            images = data_set[prevBatch1:actualBatch1]
            images    = binarize(images)
            prevBatch1=actualBatch1
            actualBatch1=(temp_samples) if (actualBatch1+batch_size>temp_samples) else (actualBatch1+batch_size)
            acc, loss = sess.run([average_accuracy, cross_entropy_loss], feed_dict={input_x: images, true_y: images[:,1:]})
            batch_accuracy += acc
            batch_loss     += loss

        # calculate the average loss and accuracy for the entire batch
        batch_average_loss     = batch_loss / total_batches
        batch_average_accuracy = batch_accuracy / total_batches
        return batch_average_accuracy, batch_average_loss

    # initialize the weights and biases
    weights = {'out': tf.Variable(tf.compat.v1.random.truncated_normal([number_hidden, number_classes]))}
    biases  = {'out': tf.Variable(tf.compat.v1.random.truncated_normal([number_classes]))}

    # placeholder for the input which is the entire image
    input_x = tf.compat.v1.placeholder(tf.float32, shape=[None, dim])

    # placeholder for the true pixel to compare except the first pixel 
    true_y = tf.compat.v1.placeholder(tf.float32, shape=[None, dim-1])

    # reshaped the image into right dimension to be passed into RNN cell
    reshaped_x = tf.reshape(input_x, [-1, number_steps, number_input])

    # compute the output of the RNN cell states and output with GRU cell
    with tf.compat.v1.variable_scope("encoder"):
        gru_cell        = tf.keras.layers.GRUCell(number_hidden)
        outputs, states = tf.compat.v1.nn.dynamic_rnn(gru_cell, reshaped_x, dtype=tf.float32)

    all_outputs = tf.reshape(outputs,[-1, number_hidden])

    # use the last rnn outputs as input and compute the output of linear layer then reshape
    linear_layer          = tf.matmul(all_outputs, weights['out']) + biases['out']
    reshaped_linear_layer = tf.reshape(linear_layer, [-1,dim])

    # remove the last output because there is nothing to compare 
    predict_y = reshaped_linear_layer[:,:-1]

    # compute the cross-entropy loss and optimize with ADAM to reduce the loss
    cross_entropy_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predict_y, labels=true_y))
    train_op           = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)

    # compute the average accuracy by counting the number of correct predicted label
    accuracy         = tf.equal(tf.compat.v1.rint(tf.nn.sigmoid(predict_y)), true_y)
    average_accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))

    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        saver             = tf.compat.v1.train.Saver()
        previous_test_loss = 10000

        # run training for 80 Epochs
        for epoch in range(number_epochs):
            prevBatch=0
            actualBatch=batch_size
            for _ in range(int(samples / batch_size)):
                images = train[prevBatch:actualBatch]
                images    = binarize(images)
                prevBatch=actualBatch
                actualBatch=(samples) if (actualBatch+batch_size>samples) else (actualBatch+batch_size)
                sess.run(train_op, feed_dict={input_x: images, true_y: images[:,1:]})

            # evaluate the loss and accuracy from both training and testing set 
            batch_train_accuracy, batch_train_loss = batch_by_batch_evaluetion(train)
            batch_test_accuracy,  batch_test_loss  = batch_by_batch_evaluetion(test)

            # display and store all the result for plotting the graph at the end of training
            print("Epoch %4d -- Training Loss: %10f -- Testing Loss: %10f -- Train Accuracy: %f -- Test Accuracy: %f" % (epoch, batch_train_loss, batch_test_loss, batch_train_accuracy*100, batch_test_accuracy*100))
            train_accuracy = np.append(train_accuracy, batch_train_accuracy * 100)
            test_accuracy  = np.append(test_accuracy, batch_test_accuracy * 100)
            train_loss     = np.append(train_loss, batch_train_loss)
            test_loss      = np.append(test_loss, batch_test_loss)

            # only save the best model when the test loss is the lowest 
            if batch_test_loss < previous_test_loss:
                previous_test_loss = batch_test_loss
                saver.save(sess, './model/gru_128')

        # plot the graph for the accuracy and loss throughout training
        x = np.linspace(0, number_epochs - 1, num=number_epochs)
        plt.figure(0)
        plt.plot(x, train_accuracy, 'r', label='Train')
        plt.plot(x, test_accuracy, 'b', label='Test')
        plt.title('Plot of Train and Test Accuracy Over %d Epochs for GRU 128 units' % (number_epochs))
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (Percent)')
        plt.legend()
        ax  = plt.subplot(111)
        box = ax.get_position()
        lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig('Plot of Train and Test Accuracy Over %d Epochs.png' % (number_epochs), bbox_extra_artists=(lgd,), bbox_inches='tight')

        plt.figure(1)
        plt.plot(x, train_loss, 'r', label='Train')
        plt.plot(x, test_loss, 'b', label='Test')
        plt.title('Plot of Train and Test Loss Over %d Epochs for GRU 128 units' % (number_epochs))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        ax  = plt.subplot(111)
        box = ax.get_position()
        lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig('Plot of Train and Test Loss Over %d Epochs.png' % (number_epochs), bbox_extra_artists=(lgd,), bbox_inches='tight')



def Predict(dataset, width, height, MODE, pixelOriginal=[]):
    #MAKE PREDICTIONS
    samples=dataset.shape[0]

    tf.compat.v1.reset_default_graph()
    number_input      = 1		
    number_steps      = 1		
    number_classes    = 1		
    number_hidden     = 128		
    number_test_image = samples		
    image_dimension   = width*height		

    # initialize the weights and biases
    weights = {'out': tf.Variable(tf.compat.v1.random.truncated_normal([number_hidden, number_classes]))}
    biases  = {'out': tf.Variable(tf.compat.v1.random.truncated_normal([number_classes]))}

    mask         = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])	   # placeholder for masked image
    reshaped_x   = tf.reshape(mask, [-1, number_steps, number_input])  # reshaped masked image into right dimension to be passed into RNN cell

    # compute the output of the RNN cell states and output with GRU cell
    with tf.compat.v1.variable_scope("encoder"):
        gru_cell        = tf.keras.layers.GRUCell(number_hidden)
        outputs, states = tf.compat.v1.nn.dynamic_rnn(gru_cell, reshaped_x, dtype=tf.float32)

    # use the last rnn outputs as input and compute the output of linear layer
    linear_layer = tf.matmul(outputs[:, -1, :], weights['out']) + biases['out']

    # store the linear layer to be used for cross entropy loss calculation with logits
    prediction = linear_layer

    # predict the first masked pixel by setting output of sigmoid to 1 if >0.5
    next_pixel = tf.reshape(tf.compat.v1.rint(tf.nn.sigmoid(linear_layer)),[1,1])

    # concat the first predicted pixel with masked image of 1 pixels to become 2 pixels
    incremental_input= tf.concat([mask, next_pixel],1)

    # predict the next masked pixels (this process is similar as above)
    with tf.compat.v1.variable_scope("encoder", reuse=True):
        for i in range(width*height-2):
            reshaped_next_pixel = tf.reshape(next_pixel, [-1, 1, number_input])

            predict_outputs, states = tf.compat.v1.nn.dynamic_rnn(gru_cell, reshaped_next_pixel, dtype=tf.float32, initial_state=states)

            linear_layer = tf.matmul(predict_outputs[:, -1, :], weights['out']) + biases['out']
            prediction   = tf.concat([prediction, linear_layer], 1)

            next_pixel        = tf.reshape(tf.compat.v1.rint(tf.nn.sigmoid(linear_layer)), [1, 1])
            incremental_input = tf.concat([incremental_input, next_pixel], 1)


    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, "./model/gru_128")

        images = dataset[:number_test_image]

        error_string=[]
        pixels=[]
        pbar= tqdm(total=100)
        precProgressBar=0
        for i in range(number_test_image):
            if(MODE=='encode'):
                im           = np.reshape(images[i] , (1,width*height))
                result = sess.run([incremental_input], feed_dict={mask: im[:, 0:1]})
            elif(MODE=='decode'):
                result = sess.run([incremental_input], feed_dict={mask: pixelOriginal[i,0:1].reshape((1,1))})
            
            # first image which is the original image
            original_image = images[i, :]
            original_image = np.reshape(original_image, (width, height))

            # second image which is the predicted image
            predicted_image = np.reshape(result,(width, height))

            if(MODE=='encode'):
                 #take all first pixels
                pixels.append(predicted_image[0][0])
            
                #create error string between two images
                columns=[]
                for j in range(width):
                    rows=[]
                    for k in range(height):
                        if(original_image[j][k] == predicted_image[j][k]):
                            rows.append(0)
                        else:
                            rows.append(original_image[j][k]-predicted_image[j][k])
                    columns.append(rows)
                error_string.append(columns)
                
            elif(MODE=='decode'):
                #create original image by errorString and predicted
                columns=[]
                for j in range(width):
                    rows=[]
                    for k in range(height):
                        rows.append(original_image[j][k]+predicted_image[j][k])
                    columns.append(rows)
                error_string.append(columns)

            valueProgressBar= int((i*100)/number_test_image)
            pbar.update(valueProgressBar-precProgressBar)
            precProgressBar=valueProgressBar

        pbar.close()
        return error_string, pixels




# in order to run code from console command: python task2a.py <arg1> <arg2> <arg3>
# arg1 = train, encode, decode
# arg2 = dataset train or encoded file (only with arg1 = decode)
# arg3 = dataset test or nothing (only with arg1 = decode)
if __name__ == '__main__':
    args = sys.argv
    sysString= "\n\n"\
            "ERROR:\n"\
            "to run the code from the console command you need to write: python DatasetCompressor.py <arg1> <arg2> <arg3>.\n"\
            "\targ1 = train\n"\
            "\targ2 = dataset train position\n"\
            "\targ3 = dataset test position\n"\
            "\t\t OR\n"\
            "\targ1 = encode\n"\
            "\targ2 = dataset train position\n"\
            "\targ3 = dataset test position\n"\
            "\t\t OR\n"\
            "you need to write: python DatasetCompressor.py <arg1> <arg2>.\n"\
            "\targ1 = encode\n"\
            "\targ2 = dataset position\n"\
            "\t\t OR\n"\
            "\targ1 = decode\n"\
            "\targ2 = encoded file position\n"

    if len(args) > 1:
        TEST_PATH=''
        if(args[1].lower() == 'train'):
            MODE='train'
            if len(args)>2:
                TEST_PATH=args[3]
            else:
                sys.exit(sysString)
        elif(args[1].lower() == 'encode'):
            MODE='encode'
            if len(args)>2:
                TEST_PATH=args[3]
        elif(args[1].lower() == 'decode'):
            MODE='decode'
        else:
            sys.exit(sysString)
        TRAIN_PATH=args[2]
    else:
        sys.exit(sysString)

    main(MODE, TRAIN_PATH, TEST_PATH)