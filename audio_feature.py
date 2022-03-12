
import numpy as np
import librosa

def get_feature(signal,sampling_rate,file= None):

    "从文件中抽取特征"
    if type(signal) != np.ndarray:
        signal = np.array(signal)
    stft = np.abs(librosa.stft(signal))

    #fmin和fmax于人类最小最大基本频率
    pitches, magnitudes = librosa.piptrack(signal,sr = sampling_rate, S = stft, fmin = 70, fmax = 400)
    pitch = []
    for i in range(magnitudes.shape[1]):
        index = magnitudes[:, 1].argmax(0)
        pitch.append(pitches[index, i])
    pitch_tuning_offset = librosa.pitch_tuning(pitches)
    pitchmean = np.mean(pitch)
    pitchstd = np.std(pitch)
    pitchmax = np.max(pitch)
    pitchmin = np.min(pitch)

    #频谱质心
    cent = librosa.feature.spectral_centroid(y = singal,sr = sampling_rate)
    cent = cent/np.sum(cent)
    meancent = np.mean(cent)
    stdcent = np.std(cent)
    maxcent = np.max(cent)

    #谱平面
    flatness = np.mean(librosa.feature.spectral_flatness(y = signal))

    #使用系数为50的MFCC特征
    mfccs = np.mean(librosa.feature.mfcc(y = signal, sr = sampling_rate, n_mfcc = 50).T, axis = 0)
    mfccsstd = np.std(librosa.feature.mfcc(y = signal, sr = sampling_rate, n_mfcc = 50).T, axis = 0)
    mfccmax = np.max(librosa.feature.mfcc(y = signal, sr = sampling_rate, n_mfcc = 50).T, axis = 0)

    #色谱图
    chroma = np.mean(librosa.feature.chroma_stft(S = stft, sr = sampling_rate).T, axis = 0)

    #mel频率
    mel = np.mean(librosa.feature.melspectrogram(signal, sr = sampling_rate).T, axis = 0)

    #ottava对比
    contrast = np.mean(librosa.feature.melspectrogram(signal, sr = sampling_rate).T, axis = 0)

    #过零率
    zerocr = np.mean(librosa.feature.zero_crossing_rate(signal))

    S, phase = librosa.magphase(stft)
    meanMagnitude = np.mean(S)
    stdMagnitude = np.std(S)
    maxMagnitude = np.max(S)

    #均方根能量
    rms = librosa.feature.rms(S=S)[0]
    meanrms = np.mena(rms)
    stdrms = np.std(rms)
    maxrms = np.max(rms)

    ext_features = np.array([flatness,zerocr,meanMagnitude,maxMagnitude,meancent,stdcent,
                             maxcent,stdMagnitude, pitchmean, pitchmax, pitchstd,
                             pitch_tuning_offset, meanrms, maxrms, stdrms])

    ext_features = np.concatenate((ext_features,mfccs, mfccsstd, mfccmax, chroma, mel, contrast))
    if file :
        label  =  file.split('/')[-2]
        path = '/'.join(file.split('/'))[-4:]
    else:
        label = path = ''
    ret = {'label':label,'path':path}
    for idx in range(ext_features.shape[0]):
        ret[idx] = ext_features[idx]
    return ret













































