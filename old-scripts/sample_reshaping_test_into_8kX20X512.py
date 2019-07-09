import numpy as np

seq_steps = 20
a = np.load("./train_input_feature_21219x512_satyajit_model.npy", allow_pickle=True)
#a = np.arange(100)
n = a.shape
print "n shape {}".format(n) 

new_len = n[0] - seq_steps + 1

b = np.zeros((new_len,20,512))
print "hello there"

for i in range(new_len):
    b[i] = a[i:i+20]

print b.shape

#
#c = np.array(b)
#print c.shape
#c = c.reshape(-1,512)
#print c.shape
np.save("train_input_feature_21200x20x512_satyajit_model.npy",b)
