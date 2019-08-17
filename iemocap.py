import os
import re
from enum import Enum
from collections import defaultdict
import xml.etree.ElementTree as ElementTree
import numpy as np

from torch.utils.data import Dataset

import pylab
import matplotlib
matplotlib.use('Agg')
import librosa
import librosa.display
import soundfile
import torch
import scipy.misc

# Audio truncation and sampling parameters
MAX_DURATION = 7.0
SAMPLE_RATE = 16000   # 16k is origin: ~60% independent, 4k: ~55% independent perf.
PADDING = int(MAX_DURATION * SAMPLE_RATE)


class Emotion(Enum):
    """ Various schemes are used when selecting emotions from IEMOCAP.
        Four basic emotions
            E, Cambria, 2017, 'Benchmarking multimodal sentiment analysis'
        Four basic emotions + excited rolled into happy
            M. Neumann, 2017, 'Attentive convolutional neural network based speech emotion recognition.'
        Four basic emotions + frustation + all other classes rolled into one class
            P. Arora, 2018, 'Exploring Siamese Neural Network Architectures for Preserving Speaker Identity.'
    Here we're using the four basic emotions to compare speaker independent performance to the benchmark paper """

    neutral = 'Neutral state'
    happy = 'Happiness'
    sad = 'Sadness'
    anger = 'Anger'
    """
    surprise = 'Surprise'
    fear = 'Fear'
    disgust = 'Disgust'
    frustration = 'Frustration'
    excited = 'Excited'
    other = 'Other'
    """


emo_int = dict()
for i, emo in enumerate(Emotion):
    emo_int[emo.name] = i


class IemocapAudio(Dataset):
    """ Load labels from 'raw' .anvil files instead of the label aggregation provided by IEMOCAP. This is because
    IEMOCAP aggregates labels according to combined motion-capture and audio-video. This has the effect of
    giving more accurate label intervals as the IEMOCAP aggregation takes the longest label interval between the two
    modalities. To compare with the benchmarking paper only labels in which at least 2 annotators agreed are kept. """
    def __init__(self, root):
        self.degrade = None
        self.data = []
        self.subject_indices = []
        idx, id = 0, 0
        for directory in os.listdir(root):
            if 'Session' in directory:
                print('Processing ', directory)
                session = SessionReader(os.path.join(root, directory, 'dialog'))

                # Hack in the subject id's for speaker recognition
                for label in session.male:
                    label.subject = id
                id += 1
                for label in session.female:
                    label.subject = id
                id += 1

                # Record the indices of each subject's samples for speaker independent validation
                self.data += session.male
                idx += len(session.male)
                self.subject_indices.append(idx)

                self.data += session.female
                idx += len(session.female)
                self.subject_indices.append(idx)

        self.data = np.array(self.data)
        self.subject_indices = np.array(self.subject_indices)
        self.len_data = len(self.data)

        print('----Dataset size', self.len_data)
        self.distribution(0, self.len_data)
        self.process_dataset()

    def process_dataset(self):
        print('Processing dataset')
        for data in self.data:
            clip, sr = data.frames[1], data.frames[2]
            spec = librosa.feature.melspectrogram(clip, sr=sr, n_mels=127)
            spec = librosa.power_to_db(spec)
            spec = np.expand_dims(spec, axis=0).astype(np.float32)
            data.add_frame(spec)

    def distribution(self, start, end):
        counts = np.zeros(len(emo_int))
        total = 0
        for i in range(start, end):
            counts[self.data[i].label] += 1
            total += 1
        counts = counts * 100 / total
        for i, emotion in enumerate(Emotion, 0):
            print(' ', emotion.name, np.round(counts[i], 2))

    def print_spectrograms(self, affix, degrade):
        for i, data in enumerate(self.data):
            if i % 200 == 0:
                if degrade is not None:
                    with torch.no_grad():
                        spectrogram = degrade(torch.Tensor([data.frames[-1]])).cpu()[0]
                else:
                    spectrogram = data.frames[-1]
                scipy.misc.imsave(data.frames[-2] + '_' + affix + '_.jpg', spectrogram[0])

    def __len__(self):
        return self.len_data

    def __getitem__(self, index):
        return self.data[index].frames[-1], self.data[index].label, self.data[index].subject


class SessionReader:
    def __init__(self, root):
        self.male = []
        self.female = []
        audio_path = os.path.join(root, 'wav')
        label_path = os.path.join(root, 'EmoEvaluation', 'Categorical')

        input_files = dict()
        for file in os.listdir(audio_path):
            # Map file name -> path
            input_files[os.path.splitext(file)[0]] = os.path.join(audio_path, file)

        # Associate input files with label files
        file_dict = defaultdict(list)
        for input_file in os.listdir(label_path):
            if '.anvil' in input_file:
                audio_name = re.search(r'^((?!(_e.*\.anvil)).)*', input_file).group()
                if audio_name in input_files:   # Only get label file if we have the associated input_file
                    audio_path = input_files[audio_name]
                    file_dict[audio_path].append(os.path.join(label_path, input_file))

        # Read label files and combine so each input file has one label object
        label_dict = dict()
        for audio_file in file_dict:
            v_label = LabelReader()
            for label_file in file_dict[audio_file]:
                v_label.add_labels(label_file)
            v_label.convert_labels()
            label_dict[audio_file] = v_label

        # Piece apart input files by label, assign label object, and create data points separated out by subject gender
        for input_file in label_dict:
            male, female = self.extract_audio_clip(input_file, label_dict[input_file])
            self.male += male
            self.female += female

    @staticmethod
    def extract_audio_clip(audio_file, v_labels):
        """ Extracts and saves clips from audio files by labelled intervals. Creates a data point for each clip. """
        dir_name = os.path.dirname(audio_file)
        audio_name = os.path.splitext(os.path.basename(audio_file))[0]
        vid_dir = os.path.join(dir_name, audio_name)
        try:
            os.mkdir(vid_dir)
        except FileExistsError:
            pass

        male, female = [], []
        for i, label in enumerate(v_labels.intervals):
            emo_count = np.zeros(len(emo_int))
            for annotation in label.annotations:
                for emotion in annotation:
                    emo_count[emo_int[emotion.name]] += 1

            if np.max(emo_count) >= 2:
                num_label = np.argmax(emo_count)
            else:
                continue

            name = audio_name + '_' + str(i) + '_' + label.gender + str(num_label)

            if name + '.wav' not in os.listdir(vid_dir):
                duration = label.end-label.start
                if duration > MAX_DURATION: duration = MAX_DURATION
                clip, sr = librosa.load(audio_file, sr=SAMPLE_RATE, offset=label.start, duration=duration)
                clip = np.pad(clip, (0, PADDING - clip.shape[0]), 'constant')
                soundfile.write(os.path.join(vid_dir, name + '.wav'), clip, sr)
            else:
                clip, sr = librosa.load(os.path.join(vid_dir, name + '.wav'), sr=None)

            dp = DataPoint(num_label)
            dp.add_frame(os.path.join(vid_dir, name + '.wav'))
            dp.add_frame(clip)
            dp.add_frame(sr)
            dp.add_frame(name)
            if label.gender == 'F':
                female.append(dp)
            else:
                male.append(dp)

        return male, female


class LabelReader:
    def __init__(self):
        self.intervals = dict()

    def add_labels(self, label_file):
        for track in ElementTree.parse(label_file).getroot().find('body').findall('track'):
            if '.Emotion' in track.attrib['name']:
                if 'Female' in track.attrib['name']:
                    gender = 'F'
                else:
                    gender = 'M'
                for el in track.findall('el'):
                    emotions = []
                    for attribute in el.findall('attribute'):
                        try:
                            emotions.append(Emotion(attribute.attrib['name']))
                        except ValueError:
                            pass    # Some categorical label files include unlisted classes
                    start = el.attrib['start']
                    end = el.attrib['end']
                    key = gender+start+end

                    if key not in self.intervals:
                        self.intervals[key] = Label(gender, start, end, emotions)
                    else:
                        self.intervals[key].annotations.append(emotions)

    def convert_labels(self):
        """ Reorders labels by time and returns as list instead of dict"""
        intervals_list = []
        for key in self.intervals:
            intervals_list.append(self.intervals[key])
        self.intervals = intervals_list
        self.intervals.sort(key=lambda x: x.start)


class Label:
    def __init__(self, gender, start, end, emotions):
        self.gender = gender
        self.start = float(start)
        self.end = float(end)
        self.annotations = []
        self.annotations.append(emotions)


class DataPoint:
    def __init__(self, label):
        self.frames = []
        self.label = label  # Not a Label object, but an int from Emotion enum
        self.subject = None

    def add_frame(self, frame_path):
        self.frames.append(frame_path)
