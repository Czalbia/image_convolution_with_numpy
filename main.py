'''
This code is a demonstration of how Sobel operator works on sample images.
By modifing the kernels, simulation of any convolution can be shown as well.

by Jan Czalbowski, 2023
'''

import numpy as np
from PIL import Image
import time

class ConvoluteValid():

    def __init__(self, name):
        self.name = name
        self.frame = Image.open(name).convert('L')
        self.feed_in_channel = np.asarray(self.frame)
        self.feed_in_channel2 = np.asarray(self.frame)
        # The first kernel (horizontal)
        self.kernel_channel = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])
        # The first kernel (vertical)
        self.kernel_channel2 = np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ])

    def quick_dot(self, array1, array2):
        return np.sum(array1 * array2)

    def convolution(self):
        stepsX = self.feed_in_channel.shape[0] - self.kernel_channel.shape[0] + 1
        stepsY = self.feed_in_channel.shape[1] - self.kernel_channel.shape[1] + 1

        new = np.zeros([stepsX, stepsY], dtype=int)
        new2 = np.zeros([stepsX, stepsY], dtype=int)

        for i in range(stepsX):
            start = time.time()
            for j in range(stepsY):
                new_input = self.feed_in_channel[i:self.kernel_channel.shape[0] + i, j:self.kernel_channel.shape[1] + j]
                new_input2 = self.feed_in_channel2[i:self.kernel_channel2.shape[0] + i,
                                                  j:self.kernel_channel2.shape[1] + j]
                new[i][j] = self.quick_dot(new_input, self.kernel_channel)
                new2[i][j] = self.quick_dot(new_input2, self.kernel_channel2)

            if i % 10 == 0:
                end = time.time()
                time_took = end - start
                print("Running...", i, "/", stepsX, "| it will take approximately:",
                      round(((stepsX - i) * time_took) / 60), "minutes to compute")

        return new, new2

    def run(self):
        output_channel, output_channel2 = self.convolution()

        mini = np.amin(output_channel)
        maxi = np.amax(output_channel)
        output_channel_prepared = 255 * ((output_channel - mini) / (maxi - mini))
        output_channel_prepared = np.round(output_channel_prepared)

        mini2 = np.amin(output_channel2)
        maxi2 = np.amax(output_channel2)
        output_channel_prepared2 = 255 * ((output_channel2 - mini2) / (maxi2 - mini2))
        output_channel_prepared2 = np.round(output_channel_prepared2)

        self.image_done = Image.fromarray(output_channel_prepared.astype('uint8'))
        self.image_done2 = Image.fromarray(output_channel_prepared2.astype('uint8'))
         # Combining the images
        self.image_done.paste(self.image_done2, (0, 0), self.image_done2)
    def save_it(self):
        new_name = "result_" + self.name
        self.image_done.save(new_name)

    def show(self):
       
        
        self.image_done.show()

# File name
image = ConvoluteValid("car_tot.png")
image.run()
image.show()
image.save_it()