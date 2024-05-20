import json
class JsonHandler:
    def __init__(self, file):
        self.file = file
    def read_json(self):
        with open(self.file, 'r') as load_f:
            file_dict = json.load(load_f)
        return file_dict

    # data is a dictionary
    def write_json(self,data):
        json_string = json.dumps(data, indent=4)
        with open(self.file, "w") as file:
            file.write(json_string)


if __name__ == '__main__':
    bgl_json = JsonHandler('../../data/multiHeadRNN/BGL_logevent2vec.json').read_json()
    hdfs_json = JsonHandler('../../data/multiHeadRNN/HDFS_logevent2vec.json').read_json()
