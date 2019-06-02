import json
import time
class Recorder:
    def __init__(self, fileName = 'data/'+time.strftime("%y%m%d%I%M%S")+'.json'):
        self.fileName = fileName
    def create(self):
        with open(self.fileName, 'w') as outfile:
            json.dump([], outfile)

    def save(self, data):
        array = self.load(self.fileName)
        array.append(data)
        with open(self.fileName, 'w') as outfile:
            json.dump(array, outfile)
    def load(self, fileName):
        with open(fileName) as jsonFile:
            data = json.load(jsonFile)
        return data
    
