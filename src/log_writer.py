import time
import matplotlib.pyplot as plt
from matplotlib import style
import os

style.use("ggplot")

model_name = "model-1570490221" # grab whichever model name you want here. We could also just reference the MODEL_NAME if you're in a notebook still.

def generate_unique_logpath(logdir, raw_run_name):
    i = 0
    while(True):
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name +".log")
        if not os.path.isfile(log_path):
            print("New log file: {}".format(log_path))
            return log_path
        i = i + 1

def write_log(log_file_path, val_acc, val_loss, train_acc, train_loss):
    with open(log_file_path, "a") as f:
        print("Logging to {}".format(log_file_path))
        f.write(f"{round(time.time(),3)},{round(float(val_acc),2)},{round(float(val_loss),4)},{round(float(train_acc),2)},{round(float(train_loss),4)}\n")

def create_acc_loss_graph(log_file_path):
    contents = open(log_file_path, "r").read().split("\n")

    list_time = []
    list_train_acc = []
    list_train_loss = []

    list_val_acc = []
    list_val_loss = []

    for c in contents:
        if "," in c:
            timestamp, val_acc, val_loss, train_acc, train_loss = c.split(",")

            list_time.append(float(timestamp))

            list_val_acc.append(float(val_acc))
            list_val_loss.append(float(val_loss))

            list_train_acc.append(float(train_acc))
            list_train_loss.append(float(train_loss))

            


    fig = plt.figure()

    ax1 = plt.subplot2grid((2,1), (0,0))
    ax2 = plt.subplot2grid((2,1), (1,0), sharex=ax1)


    ax1.plot(list_time, list_train_acc, label="train_acc")
    ax1.plot(list_time, list_val_acc, label="val_acc")
    ax1.legend(loc=2)
    ax2.plot(list_time,list_train_loss, label="train_loss")
    ax2.plot(list_time,list_val_loss, label="val_loss")
    ax2.legend(loc=2)
    plt.show()