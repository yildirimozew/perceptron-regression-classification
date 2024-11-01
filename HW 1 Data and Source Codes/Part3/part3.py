import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt

# we load all the datasets of Part 3
# the train data is already shuffled, we don't need to shuffle it...
x_train, y_train = pickle.load(open("../datasets/part3_train_dataset.dat", "rb"))
x_validation, y_validation = pickle.load(open("../datasets/part3_validation_dataset.dat", "rb"))
x_test, y_test = pickle.load(open("../datasets/part3_test_dataset.dat", "rb"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# We rescale each feature of data instances in the datasets
x_train = x_train/255.0
x_train = x_train.astype(np.float32)

x_test = x_test / 255.0
x_test = x_test.astype(np.float32)

x_validation = x_validation/255.0
x_validation = x_validation.astype(np.float32)

# and converting them into Pytorch tensors in order to be able to work with Pytorch
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train).to(torch.long)

x_validation = torch.from_numpy(x_validation)
y_validation = torch.from_numpy(y_validation).to(torch.long)

x_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test).to(torch.long)

x_train = x_train.to(device)
y_train = y_train.to(device)
x_validation = x_validation.to(device)
y_validation = y_validation.to(device)
x_test = x_test.to(device)
y_test = y_test.to(device)

num_of_hidden_layers = [1, 2]
num_of_neurons = [32, 64]
#num_of_hidden_layers = [2]
#num_of_neurons = [32]
#learning_rate = [0.001]
learning_rate = [0.01, 0.001]
activation_functions = [torch.tanh, torch.sigmoid]
#activation_functions = [torch.tanh]
#patience = [10, 30, 50]
try_amount = 10


def forward_pass(weights, biases, input_data, activation_function):
    output_layer_output = None
    w_num = 1
    hidden_layer_output = activation_function(torch.matmul(input_data, weights[0]) + biases[0])
    while(w_num < len(weights) - 1):
        hidden_layer_output = activation_function(torch.matmul(hidden_layer_output, weights[w_num]) + biases[w_num])
        w_num += 1
    #output_layer_output = torch.softmax(torch.matmul(hidden_layer_output, weights[len(weights) - 1]) + biases[len(weights) - 1], dim=1)
    output_layer_output = torch.matmul(hidden_layer_output, weights[len(weights) - 1]) + biases[len(weights) - 1]
    return output_layer_output

def init_weights(num_of_hidden_layers, num_of_neurons):
    weights = []
    biases = []
    for i in range(num_of_hidden_layers + 1):
        if(i == 0):
            weight = torch.normal(0, 1, size=(x_train.shape[1], num_of_neurons), requires_grad=True, device=device)
            weights.append(weight)
            bias = torch.normal(0, 1, size=(1, num_of_neurons), requires_grad=True, device=device)
            biases.append(bias)
        elif(i == num_of_hidden_layers):
            weight = torch.normal(0, 1, size=(num_of_neurons, 10), requires_grad=True, device=device)
            weights.append(weight)
            bias = torch.normal(0, 1, size=(1, 10), requires_grad=True, device=device)
            biases.append(bias)
        else:
            weight = torch.normal(0, 1, size=(num_of_neurons, num_of_neurons), requires_grad=True, device=device)
            weights.append(weight)
            bias = torch.normal(0, 1, size=(1, num_of_neurons), requires_grad=True, device=device)
            biases.append(bias)
    return weights, biases

def early_stop(best_accuracy, best_loss, validation_loss, validation_accuracy, counter):
    if best_accuracy == None:
        best_loss = validation_loss
        best_accuracy = validation_accuracy
        return False, best_accuracy, best_loss, counter
    
    loss_improvement = best_loss - validation_loss > 0.001
    accuracy_stagnation = abs(best_accuracy - validation_accuracy) < 0.0001
    if loss_improvement:
        best_loss = validation_loss
        best_accuracy = validation_accuracy
        counter = 0
    else:
        counter += 1

    if counter > 35 and accuracy_stagnation:
        return True, best_accuracy, best_loss, counter
    else:
        return False, best_accuracy, best_loss, counter

results = []
for hidden_layer in range(len(num_of_hidden_layers)):
    for neuron in range(len(num_of_neurons)):
        for lr in range(len(learning_rate)):
            for activation in range(len(activation_functions)):
                validation_loss_array = torch.zeros(try_amount, device=device)
                train_loss_array = torch.zeros(try_amount, device=device)
                train_accuracy_array = torch.zeros(try_amount, device=device)
                validation_accuracy_array = torch.zeros(try_amount, device=device)
                iterations_array = []
                for i in range(try_amount):
                    weights, biases = init_weights(num_of_hidden_layers[hidden_layer], num_of_neurons[neuron])
                    params = weights + biases
                    optimizer = torch.optim.Adam(params, lr = learning_rate[lr])
                    iteration = 0
                    counter = 0
                    best_accuracy = None
                    best_loss = None
                    while(True):
                        iteration += 1
                        train_predictions = forward_pass(weights, biases, x_train, activation_functions[activation])
                        train_mean_cross_entropy_loss = torch.nn.functional.cross_entropy(train_predictions, y_train)
                        optimizer.zero_grad()
                        train_mean_cross_entropy_loss.backward()
                        optimizer.step()
                        predict_label_indexes = torch.argmax(train_predictions, 1)
                        correct_predictions = (y_train == predict_label_indexes).float()
                        train_accuracy = correct_predictions.mean().item()
                        with torch.no_grad():
                            validation_predictions = forward_pass(weights, biases, x_validation, activation_functions[activation])
                            validation_mean_cross_entropy_loss = torch.nn.functional.cross_entropy(validation_predictions, y_validation) 
                            predict_label_indexes = torch.argmax(validation_predictions, 1)
                            correct_predictions = (y_validation == predict_label_indexes).float()
                            validation_accuracy = correct_predictions.mean().item()
                            if(iteration % 100 == 0):
                                print("Try: %d Hidden Layer: %d Neurons: %d Learning Rate: %.4f Activation Function: %s Iteration: %d Train Loss: %.4f Train Accuracy: %.4f Validation Loss: %.4f Validation Accuracy: %.4f" %(i + 1, num_of_hidden_layers[hidden_layer], num_of_neurons[neuron], learning_rate[lr], activation_functions[activation].__name__, iteration, train_mean_cross_entropy_loss.item(), train_accuracy, validation_mean_cross_entropy_loss.item(), validation_accuracy))
                        if(iteration > 150):
                            stop, best_accuracy, best_loss, counter = early_stop(best_accuracy, best_loss, validation_mean_cross_entropy_loss.item(), validation_accuracy, counter)
                        else:
                            stop = False
                        if(stop):
                            validation_loss_array[i] = validation_mean_cross_entropy_loss
                            train_loss_array[i] = train_mean_cross_entropy_loss
                            train_accuracy_array[i] = train_accuracy
                            validation_accuracy_array[i] = validation_accuracy
                            iterations_array.append(iteration)
                            break
                results.append({"hidden_layer": num_of_hidden_layers[hidden_layer], "neurons": num_of_neurons[neuron], "learning_rate": learning_rate[lr], "activation_function": activation_functions[activation].__name__, "train_loss": torch.mean(train_loss_array).item(), "validation_loss": torch.mean(validation_loss_array).item(), "train_accuracy": torch.mean(train_accuracy_array).item(), "validation_accuracy": torch.mean(validation_accuracy_array).item()})
                std_deviation = torch.std(validation_accuracy_array).item()
                average = torch.mean(validation_accuracy_array).item()
                confidence_interval_upper = average + 1.96 * std_deviation / np.sqrt(try_amount)
                confidence_interval_lower = average - 1.96 * std_deviation / np.sqrt(try_amount)
                f = open("../part3_results.txt", "a")
                f.write("Hidden Layer: %d Neurons: %d Learning Rate: %.4f Activation Function: %s Train Loss: %.4f Validation Loss: %.4f Train Accuracy: %.4f Validation Accuracy: %.4f\n" %(num_of_hidden_layers[hidden_layer], num_of_neurons[neuron], learning_rate[lr], activation_functions[activation].__name__, torch.mean(train_loss_array), torch.mean(validation_loss_array), torch.mean(train_accuracy_array), torch.mean(validation_accuracy_array)))
                f.write(f"Validation accuracy confidence interval is between {confidence_interval_upper} and {confidence_interval_lower}\n")
                f.close()
# we will find the best model
best_validation_accuracy = 0
best_model = None
for result in results:
    if(result["validation_accuracy"] > best_validation_accuracy):
        best_validation_accuracy = result["validation_accuracy"]
        best_model = result

# we will test the best model with the full dataset
if(best_model["activation_function"] == "tanh"):
    activation_function = torch.tanh
elif(best_model["activation_function"] == "sigmoid"):
    activation_function = torch.sigmoid
elif(best_model["activation_function"] == "relu"):
    activation_function = torch.relu
else:
    print("Activation function is not found")
iteration = 0
counter = 0
best_accuracy = None
test_loss_array = torch.zeros(try_amount, device=device)
test_accuracy_array = torch.zeros(try_amount, device=device)
for i in range(try_amount):
    weights, biases = init_weights(best_model["hidden_layer"], best_model["neurons"])
    params = weights + biases
    optimizer = torch.optim.Adam(weights + biases, lr = best_model["learning_rate"])
    iteration = 0
    counter = 0
    best_accuracy = None
    best_loss = None
    while(True):
        iteration += 1
        total_dataset = torch.cat((x_train, x_validation), 0)
        y_total = torch.cat((y_train, y_validation), 0)
        total_predictions = forward_pass(weights, biases, total_dataset, activation_function)
        total_mean_cross_entropy_loss = torch.nn.functional.cross_entropy(total_predictions, y_total) 
        optimizer.zero_grad()
        total_mean_cross_entropy_loss.backward()
        optimizer.step()
        predict_label_indexes = torch.argmax(total_predictions, 1)
        correct_predictions = (y_total == predict_label_indexes).float()
        total_accuracy = correct_predictions.mean()
        if(iteration > 15000):
            stop = True
        elif(iteration > 150):
            stop, best_accuracy, best_loss, counter = early_stop(best_accuracy, best_loss, total_mean_cross_entropy_loss.item(), total_accuracy, counter)
        else:
            stop = False
        if(iteration % 100 == 0):
            print("Iteration: %d Total Loss: %.4f Total Accuracy: %.4f" %(iteration, total_mean_cross_entropy_loss.item(), total_accuracy.item()))
        if(stop):
            break

    with torch.no_grad():
        test_predictions = forward_pass(weights, biases, x_test, activation_function)
        test_mean_cross_entropy_loss = torch.nn.functional.cross_entropy(test_predictions, y_test).item()
        predict_label_indexes = torch.argmax(test_predictions, 1)
        correct_predictions = (y_test == predict_label_indexes).float()
        test_accuracy = correct_predictions.mean()
        print("Try: %d Test Loss: %.4f Test Accuracy: %.4f" %(i, test_mean_cross_entropy_loss, test_accuracy.item()))
        test_loss_array[i] = test_mean_cross_entropy_loss
        test_accuracy_array[i] = test_accuracy

f = open("../part3_results.txt", "a")
f.write("Best Model: %s\n" %best_model)
f.write("Mean Test Loss: %.4f Mean Test Accuracy: %.4f\n" %(torch.mean(test_loss_array).item(), torch.mean(test_accuracy_array).item()))
std_deviation = torch.std(test_accuracy_array).item()
average = torch.mean(test_accuracy_array).item()
confidence_interval_upper = average + 1.96 * std_deviation / np.sqrt(try_amount)
confidence_interval_lower = average - 1.96 * std_deviation / np.sqrt(try_amount)
f.write(f"Test accuracy confidence interval is between {confidence_interval_upper} and {confidence_interval_lower}\n")
f.write("------------------------------------------------------------\n")
f.close()
