import tensorflow as tf
import keras
import config
import os
import argparse
import logging

from src import model
from src import preprocessing
from src import dataGenerator
from src import util
from src import model


physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
_config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Checking for param json folder and creating logging file
if not os.path.exists(config.pathParam):
    paramDict = config.modelParam
    util.save_json(paramDict, config.pathParam)

# Checking for logging folder and creating logging file
util.check_dir(config.pathLog)
util.set_logger(os.path.join(config.pathLog, 'train.log'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', default=None, required=False, type=str)
    parser.add_argument('--paramForce', default=False, required=False, type=bool)

    args = parser.parse_args()

    ###############################################################################
    # Loading Parameters
    assert os.path.isfile(
        config.pathParam), "No json configuration for model found at {}".format(config.pathParam)

    if args.paramForce is True:
        logging.info('Forcefully re-writing params :{}'.format(config.pathParam))
        paramDict = config.modelParam
        util.save_json(paramDict, config.pathParam)

    logging.info('Loading Parameters')
    params = util.Params(config.pathParam)

    # Training Data to Data Generator format
    logging.info('Training Data to Data Generator format')
    getTrainingData = dataGenerator.getData(pathImage=config.pathTraining)
    trainingIndexes, trainingImageMap, trainingLabelMap = getTrainingData.getList()

    logging.info('Creating Training Data Generator')
    trainingGenerator = dataGenerator.DataGenerator(
        list_IDs=trainingIndexes, imageMap=trainingImageMap, labelMap=trainingLabelMap, params=params, prediction=False, shuffle=True)

    # Validation Data to Data Generator format
    logging.info('Validation data to Data Generator format')
    getValidationData = dataGenerator.getData(pathImage=config.pathTesting)
    validationIndexes, validationImageMap, validationLabelMap = getValidationData.getList()

    logging.info('Creating Training Data Generator')
    validationGenerator = dataGenerator.DataGenerator(
        list_IDs=validationIndexes, imageMap=validationImageMap, labelMap=validationLabelMap, params=params, prediction=False, shuffle=True)

    # Initilizing Model
    logging.info('Initializing Model')

    # Model Training
    path_save_callback = os.path.join(
        config.pathOutputModel, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5')
    savingModel = keras.callbacks.ModelCheckpoint(path_save_callback,
                                                   monitor='val_loss',
                                                   verbose=0,
                                                   save_best_only=False,
                                                   save_weights_only=True,
                                                   mode='auto',
                                                   period=5)

    csvLogger = keras.callbacks.CSVLogger(
        os.path.join(config.pathLog, 'trainingLog.csv'),  append=True)

    retinaModel, retinaHeatMap = model.retinaModel(params)

    logging.info('Compiling Model')
    retinaModel.compile(optimizer=keras.optimizers.Adam(lr=params.learningRate),
                       loss='binary_crossentropy')

    retinaModel.summary()
    logging.info('Training Started...')
    try:
        history = retinaModel.fit_generator(generator=trainingGenerator,
                                           validation_data=validationGenerator,
                                           epochs=50,
                                           # sample_weight=weight_vector,
                                           use_multiprocessing=True,
                                           workers=10,
                                           callbacks=[savingModel, csvLogger]
                                           )
    except Exception as e:
        msg = 'Unable to start/complete model training. Error : {}'.format(e)
        logging.error(msg)
        raise(msg)

    logging.info('Saving Last model...')
    pathTrainedRetinaModel = os.path.join(config.pathOutputModel, 'retinaModel.h5')
    pathTrainedHeatMapModel = os.path.join(config.pathOutputModel, 'heatMapModel.h5')


    retinaModel.save(pathTrainedRetinaModel)
    retinaHeatMap.save(pathTrainedHeatMapModel)
    logging.info('Retina Model: {}, HeatMap Model: {}'.format(pathTrainedRetinaModel, pathTrainedHeatMapModel))

    