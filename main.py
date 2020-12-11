import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from functions import *
from GUI import *

physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
print("GPU is available!") if tf.config.list_physical_devices("GPU") else print("GPU is NOT available!")

#Global defines
plot = False

#Start Skript
X_train, Y_train, X_test, Y_test = load_mnist_dataset(printInfo=True)

model = build_model()

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
if __debug__:
    epochs = 1
else:
    epochs = 10

accHist = []

for i in range(7, 8):
    history = model.fit(X_train, Y_train, batch_size=2**i, epochs=epochs, verbose=2, validation_data=(X_test, Y_test))
    save_model(model, name="myModel", timestamp=True, folder="Test Batchsize", acc=history.history["val_accuracy"][-1], batch=2**i, epochs=epochs)
    accHist.append((2**i, history.history["val_accuracy"][-1]))
    print("")


print(accHist)

if plot:
    plot_metrics(history)

if __name__ == '__main__':

    if plot:
        plt.show()

    np.set_printoptions(linewidth=10000)

    app = QApplication(sys.argv)
    #m = tf.keras.models.load_model("keras_model.h5")
    #m.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    ex = MainWidget(model=model)
    # ex.showMaximized()
    ex.show()
    sys.exit(app.exec_())


