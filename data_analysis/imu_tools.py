#!/usr/bin/env python3
from scipy.fft import fft, fftfreq
import numpy as np
import matplotlib.pyplot as plt

def extractFifoData(data):
    sample_num_per_step = data["samples"][0]
    print("sample_num", sample_num_per_step)
    #Extract the data from the sensor_accel_fifo
    x = []
    y = []
    z = []
    t = []
    if sample_num_per_step == 1 and "x" in data:
        x = data["x"]
        y = data["y"]
        z = data["z"]
        t = data["timestamp"]/1e6
    else:
        for i in range(sample_num_per_step):
            x.append(data[f"x[{i}]"] * data["scale"][0])
            y.append(data[f"y[{i}]"] * data["scale"][0])
            z.append(data[f"z[{i}]"] * data["scale"][0])
            t.append(data["timestamp_sample"] + i * data["dt"])
        #Concat x data 0 1 2 3..
        x = np.concatenate([x]).T.flatten()
        y = np.concatenate([y]).T.flatten()
        z = np.concatenate([z]).T.flatten()
        t = np.concatenate([t]).T.flatten()/1e6
    xyz = np.vstack([x, y, z]).T
    #print freq 
    return t, xyz

def read_imu_ulog_fifo(data, start=0.0):
    from pyulog import ULog
    ts = []
    acc = []
    gyro = []
    t0 = None
    ulog = ULog(data)
    acc_fifo = ulog.get_dataset("sensor_accel_fifo")
    t_acc, acc = extractFifoData(acc_fifo.data)
    gyro_fifo = ulog.get_dataset("sensor_gyro_fifo")
    t_gyro, gyro = extractFifoData(gyro_fifo.data)
    #Print samples duration and freq
    print(f"Acc samples duration: {t_acc[-1] - t_acc[0]:.1f}s freq: {len(t_acc)/(t_acc[-1] - t_acc[0]):.1f}Hz")
    print(f"Gyro samples duration: {t_gyro[-1] - t_gyro[0]:.1f}s freq: {len(t_gyro)/(t_gyro[-1] - t_gyro[0]):.1f}Hz")
    imu = {
        "t_acc": t_acc,
        "acc": acc,
        "t_gyro": t_gyro,
        "gyro": gyro
    }
    return imu


def read_imu_ulog(data, start=0.0):
    from pyulog import ULog
    ts = []
    acc = []
    gyro = []
    t0 = None
    ulog = ULog(data)
    #Read sensor_accel
    acc = ulog.get_dataset("sensor_accel")
    t_acc = acc.data["timestamp"]/1e6
    acc = np.vstack([acc.data["x"], acc.data["y"], acc.data["z"]]).T
    #Read sensor_gyro
    gyro_fifo = ulog.get_dataset("sensor_gyro")
    t_gyro = gyro_fifo.data["timestamp"]/1e6
    gyro = np.vstack([gyro_fifo.data["x"], gyro_fifo.data["y"], gyro_fifo.data["z"]]).T
    #Print samples duration and freq
    print(f"Acc samples duration: {t_acc[-1] - t_acc[0]:.1f}s freq: {len(t_acc)/(t_acc[-1] - t_acc[0]):.1f}Hz")
    print(f"Gyro samples duration: {t_gyro[-1] - t_gyro[0]:.1f}s freq: {len(t_gyro)/(t_gyro[-1] - t_gyro[0]):.1f}Hz")
    imu = {
        "t_acc": t_acc,
        "acc": acc,
        "t_gyro": t_gyro,
        "gyro": gyro
    }
    return imu

def fft_data(data, fs):
    N = len(data)
    T = 1.0 / fs
    yf = fft(data)
    xf = fftfreq(N, T)[:N//2]
    return xf, 2.0/N * np.abs(yf[0:N//2])

def draw_fft_data(data, fs, label, ax, axis=""):
    xf, yf = fft_data(data, fs)
    ax.semilogy(xf, yf, label=label)
    ax.set_ylabel(f"Amp. {axis}")
    # ax.set_ylim(0, 1.2*np.max(yf))
    ax.legend()
    ax.grid()

def draw_fft(imu, name, item="acc", axs=None, figsize=(20,10)):
    if axs is None:
        plt.figure(f"{name}_fft_{item}", figsize=figsize)
        plt.clf()
        fig, axs = plt.subplots(3,1, sharex=True, figsize=figsize, num=f"{name}_fft_{item}")
    t = imu[f"t_{item}"]
    freq = len(t)/(t[-1] - t[0])
    draw_fft_data(imu[item][:,0], freq, name, axs[0], "x")
    draw_fft_data(imu[item][:,1], freq, name, axs[1], "y")
    draw_fft_data(imu[item][:,2], freq, name, axs[2], "z")
    axs[2].set_xlabel("Frequency (Hz)")
    plt.tight_layout()
    return axs
        
def draw_imu_data(imu, suffix="", axs=None, axs_gyro=None, figsize = (20,10)):
    if axs is None:
        plt.figure(f"acc", figsize=figsize)
        plt.clf()
        fig, axs = plt.subplots(3,1, sharex=True, figsize=figsize, num=f"acc")

        plt.figure(f"gyro", figsize=figsize)
        plt.clf()
        fig, axs_gyro = plt.subplots(3,1, sharex=True, figsize=figsize, num=f"gyro")
    t_acc = imu["t_acc"]
    t_gyro = imu["t_gyro"]
    axs[0].plot(t_acc, imu[f"acc{suffix}"][:,0] - np.mean(imu[f"acc{suffix}"][:,0]), label=f"a_x {suffix}")
    plt.legend()
    axs[1].plot(t_acc, imu[f"acc{suffix}"][:,1] - np.mean(imu[f"acc{suffix}"][:,1]), label=f"a_y {suffix}")
    plt.legend()
    axs[2].plot(t_acc, imu[f"acc{suffix}"][:,2], label=f"a_z {suffix}")
    plt.legend()
    axs_gyro[0].plot(t_gyro, imu[f"gyro{suffix}"][:,0], label="w_x", marker="+")
    plt.legend()
    axs_gyro[1].plot(t_gyro, imu[f"gyro{suffix}"][:,1], label="w_y")
    plt.legend()
    axs_gyro[2].plot(t_gyro, imu[f"gyro{suffix}"][:,2], label="w_z")
    return axs, axs_gyro