#Model realted
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import applications
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Sequential
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras.layers import Dropout
#graphs realted
import matplotlib.pyplot as plt

#Save results in excel sheet
import pandas

#Global Variables
img_width, img_height = 224,224
data_dir = "/content/drive/My Drive/New Dataset" #Directory to your dataset
nb_train_samples = 20713*10 #[0.9*total size of Dataset]*Data generator factor[how many images produced from ImageDataGenerator function]
nb_validation_samples = 2299*10  #[0.1*total size of Dataset]*Data generator factor[how many images produced from ImageDataGenerator function]
train_steps = 5918 # 206990 training samples/batch size of 35 = 5914 steps. We are doing heavy data processing
validation_steps = 1045 # 22980 validation samples/batch size of 20 = 1149 steps.
batch_size = 35
validation_batch_size=22
epochs = 100       #big number to make training process stop due to condition not because reach number of epochs
classes={'1_F', "1_B", '5_F', "5_B", '10_F', "10_B", '20_F', "20_B", '50_F', "50_B", '100_F', "100_B", '200_F', "200_B"} #each face of currency is class


'''
img_shape:shape of required input
preprocessing_Function: Pretrained model's preprocessing functions
'''
#generate agumented dataset
def Data_generators(img_shape, preprocessing_Function):
    '''Create the training and validation datasets for
    a given image shape.
    '''
    height, width = img_shape
    try:
        #apply some preprocessing on dataset to avoid over fitting  and generate for each image in dataset 10 images in augmented dataset
        #validation_split:getting 90% training and 10% validation
        imgdatagen = ImageDataGenerator(
            preprocessing_function=preprocessing_Function,
            rescale=1. / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.1,
            vertical_flip=True,
            validation_split=0.1,
    )
    #Get train data set from original data set 90% of total data set
        train_dataset = imgdatagen.flow_from_directory(
            data_dir,
            target_size=(height, width),
            classes=(
            '1_F', "1_B", '5_F', "5_B", '10_F', "10_B", '20_F', "20_B", '50_F', "50_B", '100_F', "100_B", '200_F', "200_B"),
            batch_size=batch_size,
            subset='training',
    )
    # Get validation data set from original data set 10% of total data set
        val_dataset = imgdatagen.flow_from_directory(
            data_dir,
            target_size=(height, width),
            classes=(
            '1_F', "1_B", '5_F', "5_B", '10_F', "10_B", '20_F', "20_B", '50_F', "50_B", '100_F', "100_B", '200_F', "200_B"),
            batch_size=validation_batch_size,
            subset='validation'
    )
        return train_dataset, val_dataset
    except:
        print("Invalid preprocessing function")
        return -1
#Generate model
def Model_generators():
    vgg16 = applications.vgg16
    train_dataset, val_dataset = Data_generators((224, 224), preprocessing_Function=vgg16.preprocess_input)
    conv_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in conv_model.layers:
        layer.trainable = False
    x = Flatten()(conv_model.output)
    # three hidden layers and add Layers to avoid over fitting
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    # final softmax layer with 14 categories (each currency face and back)
    predictions = Dense(14, activation='softmax')(x)
    # creating the full model:
    full_model = Model(inputs=conv_model.input, outputs=predictions)
    full_model.compile(loss="categorical_crossentropy",
                       optimizer=optimizers.Adamax(lr=0.001),
                       metrics=['acc'])
    checkpoint = ModelCheckpoint("/content/drive/My Drive/Egyptian_Paper_Currency_Detector_Model.h5", monitor='val_loss', verbose=1,
                                 save_best_only=True,
                                 save_weights_only=False,
                                 mode='min', period=1)
    early = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='max')
    history = full_model.fit_generator(train_dataset, shuffle=True, steps_per_epoch=train_steps, validation_data=val_dataset,
                                       validation_steps=validation_steps, workers=16, epochs=100,
                                       callbacks=[checkpoint, early])
    return history
#results graphs
def graphs(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


history=Model_generators()
graphs(history)