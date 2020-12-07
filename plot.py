def plot_images_grid_categorical(data,num_rows=1,labels=None,class_names=None):
    images, labels = data
    n=len(images)
    if n > 1:
        num_cols=np.ceil(n/num_rows)
        fig,axes=plt.subplots(ncols=int(num_cols),nrows=int(num_rows))
        axes=axes.flatten()
        fig.set_size_inches((20,20))
        for i,image in enumerate(images):
            axes[i].axis('off')
            axes[i].imshow(image.numpy())
            label = labels[i].numpy()
            axes[i].set_title(class_names[label])

def plot_images_grid(data,num_rows=1,labels=None):
    images, labels = data
    n=len(images)
    if n > 1:
        num_cols=np.ceil(n/num_rows)
        fig,axes=plt.subplots(ncols=int(num_cols),nrows=int(num_rows))
        axes=axes.flatten()
        fig.set_size_inches((20,20))
        for i,image in enumerate(images):
            axes[i].axis('off')
            axes[i].imshow(image.numpy())
            label = labels[i].numpy()
            axes[i].set_title(class_names[label])

def plot_history(history):
    fig, ax = plt.subplots(2,1)
    if history.history.get("loss"):
        [0].plot(history.history['loss'], color='b', 
            label="Training loss")
    if history.history.get("val_loss"):
        ax[0].plot(history.history['val_loss'], color='r', 
            label="Validation loss",axes =ax[0])
    legend = ax[0].legend(loc='best', shadow=True)
    if history.history.get("accuracy"):
        ax[1].plot(history.history['accuracy'], color='b', 
            label="Training accuracy")
    if history.history.get("val_accuracy"):
        ax[1].plot(history.history['val_accuracy'], color='r',
            label="Validation accuracy")
    legend = ax[1].legend(loc='best', shadow=True)