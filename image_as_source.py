import numpy as np
from PIL import Image


class convolute_valid():

    def __init__(self, path, name):
        self.image_done = 0
        self.path = path
        self.name = name
        # Importing the image as a grayscale array
        self.frame = Image.open(path + "\\" + name).convert('L')
        self.feed_in_channel = np.asarray(self.frame)
        # The kernel channel
        # Here is an example of how it looks like

        isChosen = True
        while isChosen==True:
            print('What kernel to use, press 1 for random, 2 for pre-coded one: ')
            a = input()
            if a == "1":
                self.kernel_channel = np.random.randint(100,
                                                        size=(100, 100))
                isChosen=False
            elif a == "2":
                self.kernel_channel = np.array(
                    [
                        [1, 1, 2, 2, 1, 1],
                        [1, 1, 2, 2, 1, 1],
                        [1, 2, 6, 6, 2, 1],
                        [1, 2, 6, 6, 2, 1],
                        [1, 1, 2, 2, 1, 1],
                        [1, 1, 2, 2, 1, 1]

                    ]
                )
                isChosen = False

    # My own dot product function
    def quick_dot(self, array1, array2):
        array1 = array1 * array2
        sum1 = sum(array1)
        sum2 = 0
        for i in sum1:
            sum2 += i
        return sum2

    # The convolution happens here
    def convolution(self):
        # Calculating how many steps it will take to convolute the whole image
        stepsX = self.feed_in_channel.shape[0] - self.kernel_channel.shape[0] + 1
        stepsY = self.feed_in_channel.shape[1] - self.kernel_channel.shape[1] + 1

        new = np.zeros([stepsX, stepsY], dtype=int)

        # Moving through the input channel
        lastCount =0
        for i in range(stepsX):
            for j in range(stepsY):
                # The actual conv takes place here
                # Croping the feed array so the dot product will work
                new_input = self.feed_in_channel[i:self.kernel_channel.shape[0] + i, j:self.kernel_channel.shape[1] + j]
                if (i!=lastCount):
                    print("Running...", i, "//", stepsX)
                    lastCount = i
                new[i][j] = self.quick_dot(new_input, self.kernel_channel)
        return new

    def run(self):
        # The convoluted image
        output_channel = self.convolution()

        # Converting the pixels' values of convoluted image
        # to a 0-255 gray-scale image

        mini = np.amin(output_channel)
        output_channel_prepared = output_channel
        maxi = np.amax(output_channel_prepared)
        output_channel_prepared = 255 * ((output_channel - mini) / (maxi - mini))
        output_channel_prepared = np.round(output_channel_prepared)

        # The finished image is converted to an image
        self.image_done = Image.fromarray(output_channel_prepared)

    def save(self):
        new_name = self.name+"_result.jpg"
        self.image_done.save(self.path, new_name)

    def show(self):
        self.image_done.show()


image = convolute_valid(r"C:\Users\jancz\my_stuff\pics", "IMG_5800.jpg")
image.run()
image.save()
image.show()