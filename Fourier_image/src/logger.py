import os
import sys
from shutil import copyfile
from datetime import datetime
from stat import S_IREAD, S_IRGRP, S_IROTH


class Logger:
    """
    A logger class for logging run results to a file.

    Parameters:
        file (str): The path to the file to be logged.
        dir (str, optional): The directory where the log files will be saved.
            If not provided, logs will be saved in a directory named with the current timestamp.

    Attributes:
        dir (str): The directory where the log files are saved.
        console (file): The standard output stream.
        results (file): The file object for writing log results.

    Methods:
        copyFile(file): Copies the input file to the log directory.
        createDir(dir_name): Creates a subdirectory in the log directory.
        write(message): Writes a message to both the console and the log file.
        flush(): Flushes the standard output stream.
    """

    def __init__(self, file, dir=None):
        if dir is None:
            self.dir = "./results/" + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            os.makedirs(self.dir, exist_ok=True)
        else:
            self.dir = "./" + dir

        self.copyFile(file)

        self.console = sys.stdout
        self.results = open(self.dir + "/results.txt", "a+")
        sys.stdout = self  # prints the file here
        print("LOGGING RUN RESULTS")

    def copyFile(self, file):
        copy_file = os.path.join(self.dir, os.path.basename(file))
        copyfile(file, copy_file)
        # make it so that the file can only be read.
        os.chmod(copy_file, S_IREAD | S_IRGRP | S_IROTH)

    def createDir(self, dir_name):
        os.makedirs(os.path.join(self.dir, dir_name), exist_ok=True)

    def write(self, message):
        self.console.write(message)
        self.results.write(message)

    def flush(self):
        self.console.flush()

    def __del__(self):
        self.results.close()
