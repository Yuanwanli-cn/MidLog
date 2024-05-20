import pickle
import os
class PickleHandler:
    def __init__(self, file):
        self.file = file

    def read_pickle(self, encoding='utf-8'):
        """读取pickle数据"""
        with open(self.file, 'rb') as f_read:
             data = pickle.load(f_read)
        return data

    def write_pickle(self, data, encoding='utf-8'):
        """向pickle文件写入数据"""

        if not os.path.exists(self.file):
            file_dir = os.path.dirname(self.file)
            if not os.path.isdir(file_dir):
                os.makedirs(file_dir)

        with open(self.file, 'wb') as f_save:
            pickle.dump(data, f_save)