import matplotlib.pyplot as plt
import numpy as np

#Code to Compute Sensitivity, Specificty, Precision, Recall
def compute_metrics(cm):
    # TP = np.sum((y_true == 1) & (y_pred == 1))
    # TN = np.sum((y_true == 0) & (y_pred == 0))
    # FP = np.sum((y_true == 0) & (y_pred == 1))
    # FN = np.sum((y_true == 1) & (y_pred == 0))

    TP = cm[0,0]
    TN = cm[1,1]
    FP = cm[0,1]
    FN = cm[1,0]

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)
    print('Accuracy:', accuracy)
    print('Sensitivity:', sensitivity)
    print('Specificity:', specificity)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 Score:', f1_score)

    
    return accuracy,sensitivity, specificity, precision, recall,f1_score

def plot_learning_curve(history, title='Learning Curve', xlabel='Epochs', ylabel='Loss'):
    """
    Plots the learning curve.

    Args:
        history (dict): History object returned by Keras.
        title (str): Title of the plot. Default is 'Learning Curve'.
        xlabel (str): Label of the x-axis. Default is 'Epochs'.
        ylabel (str): Label of the y-axis. Default is 'Loss'.

    Returns:
        None
    """
    # Create figure and axis
    fig, ax = plt.subplots()
    
    # Plot the learning curve
    ax.plot(history.history['loss'], label='Train')
    ax.plot(history.history['val_loss'], label='Validation')
    
    # Add legend
    ax.legend()
    
    # Set title and labels for axes
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Show the plot
    plt.show()

def plot_loss_curve(history, title='Loss Curve', xlabel='Epochs', ylabel='Loss'):
    """
    Plots the loss curve.

    Args:
        history (dict): History object returned by Keras.
        title (str): Title of the plot. Default is 'Loss Curve'.
        xlabel (str): Label of the x-axis. Default is 'Epochs'.
        ylabel (str): Label of the y-axis. Default is 'Loss'.

    Returns:
        None
    """
    # Create figure and axis
    fig, ax = plt.subplots()
    
    # Plot the loss curve
    ax.plot(history.history['loss'], label='Train')
    
    # Add legend
    ax.legend()
    
    # Set title and labels for axes
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Show the plot
    plt.show()



def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Plots the confusion matrix.

    Args:
        cm (array-like): Confusion matrix as a square array.
        classes (list or array-like): List of class labels.
        normalize (bool): Flag indicating whether to normalize the confusion matrix or not.
        title (str): Title of the plot. Default is 'Confusion Matrix'.
        cmap (matplotlib.colors.Colormap): Colormap to be used for the plot. Default is plt.cm.Blues.

    Returns:
        None
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    # Create figure and axis
    fig, ax = plt.subplots()
    
    # Plot the confusion matrix
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    
    # Set colorbar label
    cbar.ax.set_ylabel('Counts' if not normalize else 'Percentage', rotation=-90, va="bottom")
    
    # Set ticks and labels for axes
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    
    # Label the axes
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')
    
    # Loop over data dimensions and add text annotations
    for i in range(len(classes)):
        for j in range(len(classes)):
            if normalize:
                ax.text(j, i, f'{cm[i, j]:.2f}', ha="center", va="center", color="white" if cm[i, j] > 0.5 else "black")
            else:
                ax.text(j, i, f'{cm[i, j]}', ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    
    # Add title
    ax.set_title(title)
    
    # Show the plot
    plt.show()
