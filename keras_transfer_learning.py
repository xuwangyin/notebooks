
def build_model(model_name, input_shape, num_classes, transfer_learning):
    from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, Conv2D, MaxPooling2D, BatchNormalization
    from keras.models import Sequential
    from keras.applications import densenet
    from keras.applications import vgg16
    from keras.applications import InceptionV3
    from keras.models import Model

    weights = 'imagenet' if transfer_learning else None
    if model_name == 'MLP':
        model = Sequential()
        model.add(Dense(1024, activation='relu', input_shape=input_shape))
        model.add(Dropout(0.5))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
    elif model_name == 'ShallowCNN':
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=[11, 5], strides=[3, 3], input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=[3, 3], strides=[2, 2]))
        model.add(Dropout(0.2))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax'))
    elif model_name == 'VGG16':
        base_model = vgg16.VGG16(weights=weights, include_top=False, input_shape=input_shape)
        x = Flatten()(base_model.output)
        x = Dense(4096, activation='relu')(x)
        x = Dense(4096, activation='relu')(x)
        predictions = Dense(num_classes, activation="softmax")(x)
        model = Model(inputs=base_model.input, outputs=predictions)
    elif model_name == 'InceptionV3':
        base_model = InceptionV3(weights=weights, include_top=False, input_shape=input_shape)
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(4096, activation='relu')(x)
        predictions = Dense(num_classes, activation="softmax")(x)
        model = Model(inputs=base_model.input, outputs=predictions)
    elif model_name == 'DenseNet':
        model = densenet.DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
        x = Flatten()(model.output)
        predictions = Dense(num_classes, activation="softmax")(x)
        model = Model(inputs=model.input, outputs=predictions)

    return model
