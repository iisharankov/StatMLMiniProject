
import time, math, pickle

import torch
import torch.optim as optim

import matplotlib.pyplot as plt

# sys.path.append('src')
from src.models.CNN.model import Net
from src.models.CNN.data_gen import Data
import src.constants as const


def plot_results(obj_vals, cross_vals, filename):
    assert len(obj_vals) == len(cross_vals), 'Length mismatch between the curves'
    num_epochs = len(obj_vals)

    # Plot saved in results folder
    plt.plot(range(num_epochs), obj_vals, label="Training loss", color="blue")
    plt.plot(range(num_epochs), cross_vals, label="Test loss", color="green")
    plt.legend()
    plt.savefig(filename + '.pdf')
    plt.close()

def train_model(model, data):
    # Define an optimizer and the loss function
    optimizer = optim.SGD(model.parameters(), lr=const.CNN_LR)
    loss = torch.nn.BCELoss(reduction='mean')

    obj_vals = []
    cross_vals = []
    num_epochs = int(const.CNN_EPOCHS)

    # Training loop
    for epoch in range(1, num_epochs + 1):
        obj_vals.append(model.backprop(data, loss, optimizer))
        cross_vals.append(model.test(data, loss))

        # High verbosity report in output stream
        if const.CNN_V >= 2 and not ((epoch + 1) % const.CNN_DISPLAY_EPOCHS):
            train_cor, train_tot = model.accuracy(data.x_train, data.y_train)
            test_cor, test_tot = model.accuracy(data.x_test, data.y_test)
            # print(train_cor, train_tot, test_cor, test_tot)
            t = time.time() - start_time
            print('Epoch [{}/{}] - {}m{:.2f}s -'.format(epoch + 1, num_epochs, math.floor(t / 60), t % 60) +
                  '\tTraining Loss: {:.4f}  - '.format(obj_vals[-1]) +
                  'Training Accuracy: {}/{} ({:.2f}%)  -'.format(train_cor, train_tot, 100 * train_cor/train_tot) +
                  'Test Loss: {:.4f}  - '.format(cross_vals[-1]) +
                  'Test Accurecy: {}/{} ({:.2f}%)'.format(test_cor, test_tot, 100 * test_cor/test_tot))

    # Low verbosity final report
    if const.CNN_V >= 1:
        print('Final training loss: {:.4f} \t Final test loss: {:.4f}'.format(obj_vals[-1], cross_vals[-1]))
        print(cross_vals)

    return obj_vals, cross_vals

## Temp stuff for batch size i found but have not implemented yet
# loader = iter(DataLoader(TimeSeriesDataSet(x_train, y_train), batch_size=32, shuffle=True))
# # this is the chunk that goes in your training loop.
# for i in range(0, 5):
#   # next x and next y
#   x, y = loader.next()
#

if __name__ == '__main__':
    start_time = time.time()


    # Hyperparameters from json file are loaded


    # Create an instance of the model and initialize the data
    model = Net()
    data = Data(const.TRAIN_DATASET, const.TEST_DATASET)

    model_name = ""
    model_path = "savedModels/"
    load_model = ""


    # Checks if a desired model name was given, if not, creates a default one
    if model_name:
        filename = model_path + model_name
    else:
        # Replaces 2.0 as 2_0 to not clash with file naming.
        LR_as_string = str(const.CNN_LR).replace(".", "_")
        filename = model_path + f"LR-{LR_as_string}-{const.CNN_EPOCHS}Epochs"

    # checks if a model was given to load, if not, train anew
    if load_model == "":
        if const.CNN_V:
            print(f"\n Learning rate given as {const.CNN_LR}, "
                  f"with {const.CNN_EPOCHS} Epochs, and a verbosity of {const.CNN_V}\n"
                  f"No model given, training new model")

        obj_vals, cross_vals = train_model(model, data)

        # Save the model and the picked data
        torch.save(model.state_dict(), filename + ".pth")

        # Save the training and test losses
        with open(filename + ".pkl", 'wb') as pfile:
            pickle.dump((obj_vals, cross_vals), pfile)

    else:
        if const.CNN_V:
            print(f"\n Learning rate given as {const.CNN_LR}, "
                  f"with {const.CNN_EPOCHS} Epochs, and a verbosity of {const.CNN_V}\n"
                  f"Model given, attempting to load it")

        model.load_state_dict(torch.load(filename + ".pth"))
        with open(filename + ".pkl", 'rb') as pfile:
            obj_vals, cross_vals = pickle.load(pfile)

    plot_results(obj_vals, cross_vals, filename)

    # Print final loss/acceptances
    # if const.CNN_V:
    #     t = time.time() - start_time
    #     correct, total = model.accuracy(data_test.x, data_test.y)
    #     print("\nAfter training, we find the model has a test loss of {:.3f} ".format(cross_vals[-1]) +
    #           "and an accuracy of {}/{}, or {:.3f}%\n".format(correct, total, (correct/total) * 100) +
    #           "This was completed in {} minutes and {:.2f} seconds".format(math.floor(t / 60), t % 60))
