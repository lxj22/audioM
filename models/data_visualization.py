import matplotlib.pyplot as plt
import os
import argparse
import numpy as np

def data_visualization(file):
    epoch = []
    validation_loss = []
    training_loss = []
    accuracy = []
    with open(file) as f:
        for string in f:
            line = np.array(string.split()).__getitem__([2, 3, 6, 7])
            # epo = line[0]
            # epo = epo[7:epo.find('/')]
            # epoch.append(int(epo))
            validation_loss.append(float(line[1][16:]))
            training_loss.append(float(line[2]))
            accuracy.append(float(line[3][4:]))
    fig = plt.figure()
    print(training_loss)
    ax = fig.subplots(nrows=1, ncols=2)

    ax[0].plot(validation_loss, label='validation_loss')
    ax[0].plot(training_loss, label='training_loss')
    ax[0].set_title("validation_loss and training_loss")
    ax[0].legend()
    ax[1].plot(accuracy)
    ax[1].set_title("accuracy")

    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', default=os.path.join(os.getcwd(), "logs", "training_log.log"), help="Path to log file containing log directory.")

    args = parser.parse_args()

    # data visualization
    data_visualization(file=args.file)