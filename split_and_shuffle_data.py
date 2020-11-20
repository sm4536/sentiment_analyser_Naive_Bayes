import os
import shutil
import random


class Split_Data:

    def splitdata(dirname):
        source = dirname
        if not os.path.exists('training'):
            os.makedirs('training')
        if not os.path.exists('testing'):
            os.makedirs('testing')

        train = 'training'
        test = 'testing'
        files = os.listdir(source)
        random.shuffle(files)
        shuffled_list = files.copy()
        for idx, file in enumerate(shuffled_list):
            file_name = source + '/' + file
            if idx < round(len(files) * 0.8):
                shutil.copy(file_name, train)
            else:
                shutil.copy(file_name, test)
