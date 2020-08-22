import pathlib

from configure import Configure
from AI.Network.PolicyValueNet_from_junxiaosong import PolicyValueNet_from_junxiaosong
from AI.Network.PolicyValueNet_ResNet import PolicyValueNet_ResNet
import numpy as np


def select(prompt, allowed_input):
    """
    Ask the user to select an entry from the console.
    :param prompt: prompt.
    :param allowed_input: Allowed values, int type required.
    :return: <int>  Selected value.
    """
    choose_value = 1

    while True:
        input_str = input(prompt)
        try:
            input_int = int(input_str)
            if input_int in allowed_input:
                choose_value = input_int
            else:
                print("The input is incorrect. Please try again.\n")
                continue
        except:
            print("The input is incorrect. Please try again.\n")
            continue
        break
    return choose_value


def select_yes_or_no(prompt, default: bool):
    """
    Yes or No.
    Ask the user to select Yes or No.
    :param prompt: prompt.
    :param default: The default value.
    :return: <bool> Yes is True, No is False.
    """
    while True:
        input_str = input(prompt)
        if len(input_str) == 0:
            value = default
        elif input_str == "n" or input_str == "N":
            value = False
        elif input_str == "y" or input_str == "Y":
            value = True
        else:
            print("The input is incorrect. Please try again.\n")
            continue
        return value


def select_player(prompt, allowed_input):
    """
    Choose player.
    :param prompt: prompt.
    :return: (<int>, <str>)  Selected values and player names.
    """

    choose_value = select(prompt, allowed_input)

    input_name = input("It's name is.\n" ": ")
    return choose_value, input_name


def set_AI_conf(search_times=2000, greedy_value=5.0):
    """
    Set AI player configuration.
    :param search_times: Default value for AI searches.
    :param greedy_value: Default value for greedy value.
    :return: search_times, greedy_value
    """

    while True:
        input_str = input("AI Config 1: Please input the AI search times. (times >= 1)\n"
                          "AI search times ({}) = ".format(search_times))
        try:
            input_int = search_times if len(input_str) == 0 else int(input_str)
            if input_int < 1:
                print("times should be greater than or equal to 0. Please try again.\n")
                continue
            search_times = input_int
        except:
            print("The input is incorrect. Please try again.\n")
            continue
        break

    while True:
        input_str = input("AI Config 2: Please enter greedy value for Monte Carlo tree search. (c > 0)\n"
                          "c ({}) = ".format(greedy_value))
        try:
            input_float = greedy_value if len(input_str) == 0 else float(input_str)
            if input_float <= 0:
                print("c should be greater than 0. Please try again.\n")
                continue
            greedy_value = input_float
        except:
            print("The input is incorrect. Please try again.\n")
            continue
        break

    return search_times, greedy_value


def select_network(is_training=False, specified_network=0, specified_model_name=""):
    """
    Select the network you want to use.
    :param is_training: Whether to train.
    :param specified_network: Specified network.
    :param specified_model_name: Specified model.
    :return: <Network> network.
    """
    allowed_input = [1, 2]
    
    if specified_network not in allowed_input:
        network_selected = select("Please select the neural network you want to use. Press <Ctrl-C> to exit.\n"
                                  "1: Neural network provided by [junxiaosong]\n"
                                  "2: Residual Neural network\n"
                                  ": ", allowed_input=allowed_input) 
    else:
        network_selected = specified_network

    if network_selected == 1:
        is_new_model, model_dir, model_record_path = \
            select_model("Model/PolicyValueNet_from_junxiaosong", is_training, specified_model_name)
        return PolicyValueNet_from_junxiaosong(is_new_model=is_new_model, model_dir=model_dir,
                                               model_record_path=model_record_path)
    elif network_selected == 2:
        is_new_model, model_dir, model_record_path = \
            select_model("Model/PolicyValueNet_ResNet", is_training, specified_model_name)
        return PolicyValueNet_ResNet(is_new_model=is_new_model, model_dir=model_dir,
                                     model_record_path=model_record_path)


def select_model(dir: str, is_training=False, specified_model_name=""):
    """

    Select the network model you want to use or train.
    :param dir: Network model directory.
    :param is_training:  Whether to train.
    :param specified_model_name:  Specify the model you want to select.
    :return: (<bool>, <str>, <str>) 
    Whether it is a new network model, and the network model path, and the network model record path.
    """
    conf = Configure()
    conf.get_conf()
    board_conf_str = "{0}_{1}".format(conf.conf_dict["board_size"], conf.conf_dict["n_in_a_row"])
    model_path = pathlib.Path(dir)
    model_path = model_path / board_conf_str
    model_path.mkdir(parents=True, exist_ok=True)
    all_model_path = sorted(item for item in model_path.glob('*/') if item.is_dir())
    all_model_name = [path.name for path in all_model_path]

    if len(specified_model_name) != 0:
        model_path = model_path / specified_model_name
        model_path.mkdir(parents=True, exist_ok=True)
        model_record_path = model_path / "latest.h5"
        is_new_model = True
        if model_record_path.exists():
            is_new_model = False
        return is_new_model, str(model_path) + "/", str(model_record_path)

    if is_training:
        print("Please select the network model you want to train. Press <Ctrl-C> to exit.")
        print("0: Create a new network model.")
    else:
        print("Please select the network model you want to use. Press <Ctrl-C> to exit.")
    for i, one_model_name in enumerate(all_model_name):
        print("{0}: {1}".format(i + 1, one_model_name))

    model_selected = select(": ", allowed_input=range(0 if is_training else 1, len(all_model_path) + 1))
    if model_selected == 0:
        while True:
            new_name = input("Please enter a new model name. Press <Ctrl-C> to exit.\n"
                             ": ") + str(np.random.rand())
            if len(new_name) == 0:
                print("Model name is empty, please try again.\n")
                continue
            if new_name in all_model_name:
                print("The model name already exists, please try again.\n")
                continue
            model_path = model_path / new_name
            model_path.mkdir(parents=True, exist_ok=True)
            return True, str(model_path) + "/", None
    else:
        model_path = all_model_path[model_selected - 1]
        model_record_path = sorted(item for item in model_path.glob('*.h5'))
        model_record_name = [path.name[:-3] for path in model_record_path]
        if is_training:
            print("Please select the model record you want to train. Press <Ctrl-C> to exit.")
        else:
            print("Please select the model record you want to use. Press <Ctrl-C> to exit.")
        for i, one_model_record_name in enumerate(model_record_name):
            print("{0}: {1}".format(i + 1, one_model_record_name))
        model_record_selected = select(": ", allowed_input=range(1, len(model_record_path) + 1))
        return False, str(model_path) + "/", str(model_record_path[model_record_selected - 1])
