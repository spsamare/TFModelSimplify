from keras.models import load_model


if __name__ == '__main__':
    # load the uncompressed keras model
    model = load_model('models/uncompressed.hdf5')
    # serialize it
    model_json = model.to_json()
    # save the serialized structure
    with open('models/compressed.json', 'w') as json_file:
        json_file.write(model_json)
    # save the weights
    model.save_weights('models/compressed.hdf5')
