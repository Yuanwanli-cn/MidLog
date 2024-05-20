import yaml
import os
class YamlHandler:
    def __init__(self, file):
        self.file = file

    def read_yaml(self, encoding='utf-8'):
        """读取yaml数据"""
        with open(self.file, encoding=encoding) as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)

    def write_yaml(self, data,encoding='utf-8'):
        """向yaml文件写入数据"""

        if not os.path.exists(self.file):
            file_dir = os.path.dirname(self.file)
            if not os.path.isdir(file_dir):
                os.makedirs(file_dir)
            open(self.file,'w').close()

        with open(self.file, encoding=encoding, mode='w') as f:
            return yaml.dump(data, stream=f, allow_unicode=True)