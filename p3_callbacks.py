from keras.callbacks import Callback
import matplotlib.pyplot as plt

# Callback to stop early based on a value of loss
class EarlyStoppingByLossVal(Callback):
    def __init__(self, value=0.05):
        self.value = value
        
    def on_epoch_end(self, epoch, logs={}):
        current = logs.get('val_loss')
        if current < self.value:
            print("Epoch %05d: Stopping as val loss threshold is reached" % epoch)
            self.model.stop_training = True

# Callback to plot validation and training losses
class PlotCallback(Callback):
    def __init__(self):
        plt.ion()
        self.ep_list = []
        self.tr_loss = []
        self.va_loss = []
        
    def on_epoch_end(self, epoch, logs):
        plt.clf()
        self.ep_list.append(epoch+1)
        self.tr_loss.append(logs['loss'])
        self.va_loss.append(logs['val_loss'])
        plt.plot(self.ep_list,self.tr_loss,'bo-',label='Training Loss')
        plt.plot(self.ep_list,self.va_loss,'rx-',label='Validation Loss')
        plt.title("Training and validation losses (MAE)")
        plt.xlabel("Number of epochs")
        plt.ylabel("Absolute loss")
        plt.legend(loc='best')    
        plt.draw()
        plt.pause(1e-10)     

    def on_train_end(self,logs):
        plt.show()