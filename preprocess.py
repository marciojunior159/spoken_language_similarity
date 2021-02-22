import pandas as pd
import numpy as np
from librosa.feature import mfcc, delta
from pydub import AudioSegment
from pydub.playback import play
from glob import glob
import os

cwd = os.getcwd()
print(cwd)

# Variable to store datasets for each spoken language
data = {}

# Iterate over all folders to collect usable data, transforming as it goes to save memory
for folderpath in glob(r"data\*"):
    if '_' not in folderpath:
        train_data = pd.read_csv(folderpath + r"\validated.tsv", sep='\t', ).head(1000)

        # print(train_data.iloc[0])

        mfcc_data = pd.DataFrame()
        nn = 1
        for clip in train_data['path']:
            try:
                fullpath = r"{}\{}\clips\{}".format(cwd, folderpath, clip)
                audio = AudioSegment.from_mp3(fullpath)
                features = delta(mfcc(np.float64(np.array(audio.get_array_of_samples()))[:196000], sr=audio.frame_rate, n_fft=1024, hop_length=1024), width=9, mode='interp')
                print(folderpath, features.shape, np.array(audio.get_array_of_samples()).shape, nn)
                data[clip[:-4]] = {'lang':clip.split('_')[2], 'features':features.tolist()}
                nn += 1
            except:
                print("None")

import json
with open('data.json', 'w') as fp:
    json.dump(data, fp)
            


