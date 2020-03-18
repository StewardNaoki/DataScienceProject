import time
import matplotlib.pyplot as plt
from matplotlib import style
import os
import sys

style.use("ggplot")

# grab whichever model name you want here. We could also just reference the MODEL_NAME if you're in a notebook still.
model_name = "model-1570490221"


class LogManager:
    def __init__(self, logdir, raw_run_name):
        self.logdir = logdir
        self.raw_run_name = raw_run_name
        self.run_num = 0
        self.example_text = ""
        self.num_image_train = 0
        self.num_image_test = 0
        self.max_image = 10

    def set_tensorboard_writer(self, tensorboard_writer):
        self.tensorboard_writer = tensorboard_writer

    def generate_unique_dir(self):
        i = 0
        while(True):
            # i = int(time.time() % MAX_TIME)
            i = int(time.time())
            run_name = self.raw_run_name + str(i)
            run_folder = os.path.join(self.logdir, run_name)
            if not os.path.isdir(run_folder):
                print("New run folder: {}".format(run_folder))
                os.mkdir(run_folder)
                self.run_num = i
                self.run_dir_path = run_folder
                return run_folder + "/", i
            # i = i + 1
            time.sleep(1)

    def generate_unique_logpath(self):
        i = 0
        while(True):
            # i = int(time.time() % MAX_TIME)
            i = int(time.time())
            run_name = self.raw_run_name + str(i)
            log_path = os.path.join(self.logdir, run_name + ".log")
            if not os.path.isfile(log_path):
                print("New log file: {}".format(log_path))
                return log_path
            time.sleep(1)

    def write_log(self, log_file_path, val_acc, val_loss, train_acc, train_loss):
        with open(log_file_path, "a") as f:
            print("Logging to {}".format(log_file_path))
            f.write(f"{round(time.time(),3)},{round(float(val_acc),2)},{round(float(val_loss),4)},{round(float(train_acc),2)},{round(float(train_loss),4)}\n")

    def tensorboard_send_image(self, index, image_input, mask_target, mask_output, txt="testing"):
        if txt == "training":
            self.num_image_train += 1
            print("sending image {} {}".format(txt, self.num_image_train))
        elif txt == "testing":
            self.num_image_test += 1
            print("sending image {} {}".format(txt, self.num_image_test))
        if self.num_image_test > self.max_image:
            return
        if self.num_image_test > self.max_image:
            return
        self.tensorboard_writer.add_image(
            '{}_image/{}'.format(txt, index), image_input, 0)
        self.tensorboard_writer.add_image(
            '{}_mask_target/{}'.format(txt, index), mask_target, 0)
        self.tensorboard_writer.add_image(
            '{}_mask_output/{}'.format(txt, index), mask_output, 0)

    def summary_writer(self, model, optimizer):

        summary_file = open(self.run_dir_path + "/summary.txt", 'w')

        summary_text = """
RUN Number: {}
===============

Executed command
================
{}

Model summary
=============
{}


{} trainable parameters

Optimizer
========
{}
""".format(self.run_num, " ".join(sys.argv), model, sum(p.numel() for p in model.parameters() if p.requires_grad), optimizer)
        summary_file.write(summary_text)
        summary_file.close()


# def generate_unique_run_dir(logdir, raw_run_name):
#     i = 0
#     while(True):
#         run_name = raw_run_name + "_" + str(i)
#         run_folder = os.path.join(logdir, run_name)
#         if not os.path.isdir(run_folder):
#             print("New run folder: {}".format(run_folder))
#             return run_folder, i
#         i = i + 1

# def generate_unique_logpath(logdir, raw_run_name):
#     i = 0
#     while(True):
#         run_name = raw_run_name + "_" + str(i)
#         log_path = os.path.join(logdir, run_name +".log")
#         if not os.path.isfile(log_path):
#             print("New log file: {}".format(log_path))
#             return log_path
#         i = i + 1

# def write_log(log_file_path, val_acc, val_loss, train_acc, train_loss):
#     with open(log_file_path, "a") as f:
#         print("Logging to {}".format(log_file_path))
#         f.write(f"{round(time.time(),3)},{round(float(val_acc),2)},{round(float(val_loss),4)},{round(float(train_acc),2)},{round(float(train_loss),4)}\n")

# def create_acc_loss_graph(log_file_path):
#     contents = open(log_file_path, "r").read().split("\n")

#     list_time = []
#     list_train_acc = []
#     list_train_loss = []

#     list_val_acc = []
#     list_val_loss = []

#     for c in contents:
#         if "," in c:
#             timestamp, val_acc, val_loss, train_acc, train_loss = c.split(",")

#             list_time.append(float(timestamp))

#             list_val_acc.append(float(val_acc))
#             list_val_loss.append(float(val_loss))

#             list_train_acc.append(float(train_acc))
#             list_train_loss.append(float(train_loss))

#     fig = plt.figure()

#     ax1 = plt.subplot2grid((2,1), (0,0))
#     ax2 = plt.subplot2grid((2,1), (1,0), sharex=ax1)


#     ax1.plot(list_time, list_train_acc, label="train_acc")
#     ax1.plot(list_time, list_val_acc, label="val_acc")
#     ax1.legend(loc=2)
#     ax2.plot(list_time,list_train_loss, label="train_loss")
#     ax2.plot(list_time,list_val_loss, label="val_loss")
#     ax2.legend(loc=2)
#     plt.show()
