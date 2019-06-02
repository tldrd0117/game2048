from mcts.record.Recorder import Recorder
import numpy as np
recorder = Recorder()
data = recorder.load('data/190601054701.json')
print(data)

print(max(data, key=lambda item: item['value']))
    
