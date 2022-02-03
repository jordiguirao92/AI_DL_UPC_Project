import matplotlib.pyplot as plt

def get_plot_loss(loss_train, loss_val):
    plt.title("Training&Validation loss")
    plt.plot(loss_train, label='train')
    plt.plot(loss_val, label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


