import torch
import torchaudio

import numpy as np

class Audio_Parser():
    def __init__(self):
        super(Audio_Parser, self).__init__()
        print("init Audio_Parser")

    def load_audio(self, audio_path):
        '''
        waveform: [num_channels, num_frames]
        sample_rate: 每秒钟采样点数, 48kHz表明每秒钟采样48000
        duration: 持续时长(seconds)
        '''
        waveform, sample_rate = torchaudio.load(audio_path)
        duration = waveform.shape[1] / sample_rate # seconds
        return waveform, sample_rate, duration
    
    def aligh_fps(self, waveform, sample_rate, video_fps = 30):
        samples_per_frame = int(sample_rate / video_fps) # 每帧视频对应的音频采样数

        audio_segments = []
        for start in range(0, waveform.shape[1], samples_per_frame):
            end = start + int(samples_per_frame)
            audio_segments.append(waveform[:, start:end])

        if(audio_segments[-1].shape[1] != samples_per_frame): # 填充最后一帧
            pad_val = torch.zeros(1, samples_per_frame - audio_segments[-1].shape[1])
            audio_segments[-1] = torch.concat((audio_segments[-1], pad_val), dim = 1)

        aligh_audio = torch.zeros(len(audio_segments), waveform.shape[0], samples_per_frame)
        for idx in range(len(audio_segments)):
            aligh_audio[idx] = audio_segments[idx]

        return aligh_audio # T C samples_per_frame

if __name__ == "__main__":
    test_audio = './48114_超自然梓梓.mp3'

    parser = Audio_Parser()
    audio, sample_rate, duration = parser.load_audio(test_audio) # sample_rate 24kHZ
    aligh_audio = parser.aligh_fps(audio, sample_rate, 30) # 30 fps
    print("All done!")