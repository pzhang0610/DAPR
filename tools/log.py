import os
import os.path as osp


class Logger(object):
    def __init__(self, logger_path=None):
        if logger_path is None:
            self.logger_path = './logs/log.txt'
        else:
            self.logger_path = logger_path

        dir_path = osp.dirname(self.logger_path)

        if not osp.exists(dir_path):
            os.makedirs(dir_path)
        self.file = open(self.logger_path, 'w')

    def __del__(self):
        self.close()

    def __exit__(self, *args):
        self.close()

    def __call__(self, msg, newline=True, sysout=True):
        msg = str(msg)
        if newline:
            msg += '\n'

        with open(self.logger_path, 'a') as f:
            f.write(msg)
            f.close()

        if sysout:
            print(msg)

    def close(self):
        if self.file is not None:
            self.file.close()


