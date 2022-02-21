import matplotlib.pyplot as plt
from utils.metrics import get_history_metrics

def get_plot_loss():
    loss_history_train, loss_history_val = get_history_metrics('loss')
    plt.title("Training&Validation loss")
    plt.plot(loss_history_train, label='train')
    plt.plot(loss_history_val, label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


