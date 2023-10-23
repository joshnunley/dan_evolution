import numpy as np
import gym
import pygad
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#TODO: 
# - Add abstract graph defining the layers, node properties and edge properties of the dan model
# - Define a mapping function that maps the genes to phenotype (model parameters)
# - Add the model to the fitness function, making sure it is properly accepting inputs and returning outputs




def fitness_func(solution, sol_idx):
    global model, env

    model.set_weights(pygad.kerasga.model_weights_as_matrix(model=model,
                                                            weights_vector=solution))

    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        state = np.reshape(state, (1, 4))
        q_values = model.predict(state)
        action = np.argmax(q_values[0])

        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state

    return total_reward

# Initialize CartPole environment
env = gym.make("CartPole-v1")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

hidden_state_size = 30
n_evals = 15
graph_type = 'basic_fully_connected_input'
properties_for_nodes, properties_for_edges, output_id = get_abstract_graph(graph_type, device)
print('graph_type: ', graph_type, "number of evals: ", n_evals)


learn_initial_hidden_state = False
model = DynamicAttentionNetwork(
    abstract_graph=abstract_graph,
    hidden_state_size=hidden_state_size,
    num_evals=n_evals,
    attention_bounding_method="cosine",
    learn_initial_hidden_state=learn_initial_hidden_state,
    device=device,
).to(device)

# Initialize GA parameters
num_solutions = 50
num_parents_mating = 10
num_generations = 100
mutation_percent_genes = 5

# Initialize GA instance
ga_instance = pygad.GA(
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=fitness_func,
    sol_per_pop=num_solutions,
    num_genes=model.count_params(),
    init_range_low=-1,
    init_range_high=1,
    mutation_percent_genes=mutation_percent_genes,
)

# Run the GA
ga_instance.run()

# After the generations complete, you can extract the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Best solution fitness value: {solution_fitness}")

# You can also set the neural network weights to the best solution.
model.set_weights(pygad.kerasga.model_weights_as_matrix(model=model,
                                                        weights_vector=solution))


import torch
import networkx as nx
import numpy as np
from copy import deepcopy

from DynamicAttentionNetwork import DynamicAttentionNetwork
from graph_structures import get_abstract_graph




print(
    "Parameter Count: {}".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )
)


# print the prameters and their names
for name, param in model.named_parameters():
    print(name, param.shape)

# train the Dynamic Attention Network
test_accuracy = -1
optimizer.zero_grad()
for epoch in range(30):
    partial_train_accuracy = 0
    train_correct = 0
    for i, (images, labels) in enumerate(mnist_train_loader):
        labels = labels.to(device)
        input_states = {}
        input_states[0] = images.view(batch_size, 28 * 28, 1).to(device)

        if learn_initial_hidden_state:
            outputs, hidden_states = model(
                input_states
            )
        else:
            outputs, hidden_states = model(
                input_states, deepcopy(initial_hidden_states)
                )
        

        for j in range(n_evals):
            outputs, hidden_states = model(input_states, hidden_states)

        output = outputs[output_id][:, :, 0]

        # TODO: Add an output scale parameter to every node that
        # has an output.
        loss = F.nll_loss(F.log_softmax(100*output, dim=1), labels) / num_gradient_accumulations

        train_correct += (output.argmax(dim=1) == labels).sum().item()
        partial_train_accuracy = train_correct / ((i + 1) * batch_size)

        loss.backward()
        if (((i + 1) % num_gradient_accumulations) == 0) or (i + 1 == len(mnist_train_loader)):
            optimizer.step()
            optimizer.zero_grad()

        # for name, param in dan.named_parameters():
        #    print(name, param.grad)
        if (i + 1) % (100*num_gradient_accumulations) == 0:
            print(
                "Epoch: {}, Batch: {}, Loss: {}, Partial Train Accuracy: {}, Test Accuracy: {}".format(
                    epoch, int(i/num_gradient_accumulations), loss.item()*num_gradient_accumulations, partial_train_accuracy, test_accuracy
                )
            )
        if (i + 1) % 1000 == 0:
            with torch.no_grad():
                # calculate the accuracy of the model on a single batch
                # of the test data
                # store a time series of the relevancies
                relevancy_test_size = 10
                relevancy_time_series = []
                hidden_state_time_series = []

                correct = 0
                total = 0
                images, labels = next(iter(mnist_test_loader))
                labels = labels[:relevancy_test_size].to(device)
                input_states[0] = (
                    images[:relevancy_test_size]
                    .view(relevancy_test_size, 28 * 28, 1)
                    .to(device)
                )

                if learn_initial_hidden_state:
                    outputs, hidden_states = model(
                        input_states
                    )
                else:
                    outputs, hidden_states = model(
                        input_states, deepcopy(initial_hidden_states)
                        )
                hidden_state_time_series.append(hidden_states)
                relevancy_time_series.append(model.relevancies)
                for j in range(n_evals):
                    outputs, hidden_states = model(input_states, hidden_states)
                    hidden_state_time_series.append(hidden_states)
                    relevancy_time_series.append(model.relevancies)

                output = outputs[output_id][:, :, 0]
                predicted = torch.argmax(output, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                partial_test_accuracy = correct / total

                # checkpoint the model
                torch.save(
                    model.state_dict(),
                    "./saved_data/checkpoint_dan"
                    + str(hidden_state_size)
                    + "_"
                    + str(n_evals)
                    + "_"
                    + graph_type
                    + "_classify.pt",
                )
                # save the relevancy time series
                torch.save(
                    relevancy_time_series,
                    "./saved_data/relevancy_time_series"
                    + str(hidden_state_size)
                    + "_"
                    + str(n_evals)
                    + "_"
                    + graph_type
                    + "_classify.pt",
                )
                torch.save(
                    hidden_state_time_series,
                    "./saved_data/hidden_state_time_series"
                    + str(hidden_state_size)
                    + "_"
                    + str(n_evals)
                    + "_"
                    + graph_type
                    + "_classify.pt",
                )
                torch.save(
                    images,
                    "./saved_data/test_images"
                    + str(hidden_state_size)
                    + "_"
                    + str(n_evals)
                    + "_"
                    + graph_type
                    + "_classify.pt",
                )

    with torch.no_grad():
        print("Calculating accuracy on entire test set...")
        # Print the accuracy of the model on the entire test dataset
        correct = 0
        total = 0
        for images, labels in mnist_test_loader:
            labels = labels.to(device)
            input_states[0] = images.view(test_batch_size, 28 * 28, 1).to(device)

            if learn_initial_hidden_state:
                outputs, hidden_states = model(
                    input_states
                )
            else:
                outputs, hidden_states = model(
                    input_states, deepcopy(initial_hidden_states)
                    )
            for j in range(n_evals):
                outputs, hidden_states = model(input_states, hidden_states)

            output = outputs[output_id][:, :, 0]
            predicted = torch.argmax(output, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        test_accuracy = correct / total
        print(test_accuracy)

    scheduler.step()

# print all the model parameters
for name, param in model.named_parameters():
    print(name, param)

# Print the parameter count
print(
    "Parameter Count: {}".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )
)
print("Test Accuracy: {}".format(test_accuracy))