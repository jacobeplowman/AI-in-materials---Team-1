def simple_nn(X,y):

    import tensorflow
    from tensorflow import keras
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPRegressor

    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[223]))
    model.add(keras.layers.Dense(300, activation="relu"))
    model.add(keras.layers.Dense(100, activation="relu"))
    model.add(keras.layers.Dense(10, activation="softmax"))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
    return regr.score(X_test, y_test)
