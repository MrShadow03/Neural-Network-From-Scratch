'''
Implementation of Categorical Cross Entropy loss calculation function
'''
import math
import numpy as np

softmax_output = [0.7, 0.1, 0.2]
target_output = [1, 0, 0]

loss = -(
    target_output[0]*math.log(softmax_output[0])+
    target_output[1]*math.log(softmax_output[1])+
    target_output[2]*math.log(softmax_output[2])
)

# can be further simplified to -log(predicted output)
loss = -math.log(softmax_output[0])

# here is the calculation when the output is in batch
softmax_outputs = [[0.7, 0.1, 0.2],
                    [0.1, 0.5, 0.4],
                    [0.02, 0.9, 0.08]]

class_targets = [0, 1, 1]
#as target outputs for batch
target_outputs = [[1, 0, 0],
                  [0, 1, 0],
                  [0, 1, 0]]

#using zipping and list comprehension
for index, distribution in zip(class_targets, softmax_outputs):
    print(distribution[index])
    
#using 0 multiplication
pred_list = np.sum((np.array(softmax_outputs) * np.array(target_outputs)), axis=1)
print(-np.log(pred_list))
   
#using numpy
pred_list = np.array(softmax_outputs)[range(len(softmax_outputs)), class_targets]
print(np.mean(-np.log(pred_list)))
