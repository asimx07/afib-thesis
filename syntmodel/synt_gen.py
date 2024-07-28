import numpy as np
from syntmodel.signal_model import gen_rr, gen_phase, gen_ecg
from syntmodel.noise_model import psd2time


def model(n, prms, fs,rr=None):
    '''
    Returns one ecg witn n r-peaks and corresponding noise stream
    and r_indices.
    '''
    if rr is None:
        rr = gen_rr(n, prms)
    else:
        rr = rr
    rr_phase = gen_phase(rr, fs)
    time, ecg_clean, label = gen_ecg(rr_phase, fs, prms=prms)
    _, _, _, noise_stream = psd2time(len(ecg_clean), prms, fs)
    return ecg_clean, noise_stream, label



def generate_randomset(args, prms):
    '''
    Generates a random set of ecgs where args (dict) holds generation input
    arguments and prms the signal properties..

    '''

    sample_len = args.syntdata.fs*args.data.sample_time

    dataset, noiseset = [], []
    for i in range(args.data.n_samples):
        prms.randomize()
        #n_rri = 10, generate enough extra so it can be cropped to desired length
        ecg_clean, noise_stream, (p,q,r,s,t) = model(10, prms, args.syntdata.fs)

        all_ = (ecg_clean, (p,q,r,s,t), noise_stream)
        data, noise_stream = crop_signal(all_, sample_len, args.syntdata.fs, prms.mu)
        dataset.append(data)
        noiseset.append(noise_stream)

    return dataset, noiseset


def generate_sinus_randomset(params, fs, sample_time, n_samples):
    '''
    Generates a random set of ECGs

    Args:
        params (Prms): The Prms object used to control signal properties.
        fs (int): The sampling frequency.
        sample_time (float): The duration of each sample in seconds.
        n_samples (int): The number of samples to generate.

    Returns:
        dataset (list): List of generated ECG datasets.
        noiseset (list): List of generated noise streams.
    '''
    sample_len = fs * sample_time

    dataset, noiseset = [], []
    for i in range(n_samples):
        params.randomize()
        # n_rri = 10, generate enough extra so it can be cropped to desired length
        ecg_clean, noise_stream, indices = model(15, params, fs)

        all_ = (ecg_clean, indices, noise_stream)
        data, noise_stream = crop_signal(all_, sample_len, fs, params.mu)
        dataset.append(data)
        noiseset.append(noise_stream)

    return dataset, noiseset


def generate_afib_randomset(params, fs, sample_time, n_samples, synth_RR):
    '''
    Generates a random set of ECGs

    Args:
        params (Prms): The Prms object used to control signal properties.
        fs (int): The sampling frequency.
        sample_time (float): The duration of each sample in seconds.
        n_samples (int): The number of samples to generate.

    Returns:
        dataset (list): List of generated ECG datasets.
        noiseset (list): List of generated noise streams.
    '''
    sample_len = fs * sample_time

    dataset, noiseset = [], []
    for i in range(n_samples):
        params.randomize()
        # n_rri = 10, generate enough extra so it can be cropped to desired length
        ecg_clean, noise_stream, indices = model(20, params, fs,synth_RR)

        all_ = (ecg_clean, indices, noise_stream)
        data, noise_stream = crop_signal(all_, sample_len, fs, params.mu)
        dataset.append(data)
        noiseset.append(noise_stream)

    return dataset, noiseset

def crop_signal(signal, wlen, fs, random_start):
    '''
    Crops a signal and r-indices into spesific length (wlen) and

    '''
    #use mu fron synt model for starting randomization
    ecg, labels, noise_stream = signal
    start = np.random.randint(0, int(fs*random_start))
    stop = start + wlen
    assert (len(ecg) > stop)
    ecg = ecg[start:stop]
    noise_stream = noise_stream[start:stop]

    p,q,r,s,t = labels
    p = p[(p >= start) & (p <= stop-1)] - start
    q = q[(q >= start) & (q <= stop-1)] - start
    r = r[(r >= start) & (r <= stop-1)] - start
    s = s[(s >= start) & (s <= stop-1)] - start
    t = t[(t >= start) & (t <= stop-1)] - start

    return (ecg, p,q,r,s,t), noise_stream


