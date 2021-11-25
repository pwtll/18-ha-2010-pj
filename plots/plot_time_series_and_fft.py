import scipy.io
from scipy.fftpack import fft, fftfreq
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os
import numpy as np

path_to_csv = r"C:\Users\Philipp Witulla\PycharmProjects\training"
Tk().withdraw()
file_directory = askopenfilename(initialdir=path_to_csv, filetypes =((".mat File", "*.mat"),(".csv File", "*.csv"),("All Files","*.*")))
filename = os.path.basename(file_directory)

data = scipy.io.loadmat(file_directory)

fig, (ax1, ax2) = plt.subplots(2, sharex=False, sharey=False, figsize=(12,6))
fig.suptitle('ECG time series and FFT of ' + filename, fontsize=12)
params = {'mathtext.default': 'regular',
          'axes.titlesize': 16,
          'axes.labelsize': 14,
          'font.family' : 'sans-serif',
          'font.sans-serif' : 'Tahoma'
          }        # mathematische Achsenbeschriftungen
plt.rcParams.update(params)

number_of_data_points = range(data['val'].size)
time = []
for data_point in number_of_data_points:
    time.append(data_point/300)            # 300 Hz sampling Frequency

ax1.plot(time, data['val'][0])        # ax1.plot(time[1500:], data['val'][0][1500:])
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('ECG Amplitude')
ax1.grid(b=True, which='both', axis='y')

# Number of sample points
N = data['val'].size    # -1500
# sample spacing
T = 1.0 / 300.0
x = np.linspace(0.0, N*T, N, endpoint=False)

yf = fft(data['val'][0])    # fft(data['val'][0][1500:])
xf = fftfreq(N, T)[:N//2]

ax2.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('FFT Amplitude')
plt.grid(b=True, which='both', axis='y')
ax2.set_xscale('log')

fig.tight_layout()
plt.show()







LCDML_add         (0 , LCDML_0         , 1  , "Temperatures", NULL);       // NULL = no menu function
LCDML_add         (1 , LCDML_0_1       , 1  , "", mFunc_back);
LCDML_addAdvanced (2 , LCDML_0_1       , 2  , NULL              ,          ""      , showTemps,       1,            _LCDML_TYPE_dynParam);
LCDML_addAdvanced (3 , LCDML_0_1       , 3  , NULL              ,          ""      , showTemps,       2,            _LCDML_TYPE_dynParam);
LCDML_addAdvanced (4 , LCDML_0_1       , 4  , NULL              ,          ""      , showTemps,       3,            _LCDML_TYPE_dynParam);
LCDML_addAdvanced (5 , LCDML_0_1       , 5  , NULL              ,          ""      , showTemps,       4,            _LCDML_TYPE_dynParam);
LCDML_add         (6 , LCDML_0_1       , 6  , "Show more"            , NULL);
LCDML_add         (7 , LCDML_0_1_6     , 1  , "", mFunc_back);
LCDML_addAdvanced (8 , LCDML_0_1_6     , 2  , NULL              ,          ""      , showTemps2,       1,            _LCDML_TYPE_dynParam);
LCDML_addAdvanced (9 , LCDML_0_1_6     , 3  , NULL              ,          ""      , showTemps2,       2,            _LCDML_TYPE_dynParam);
LCDML_addAdvanced (10 , LCDML_0_1_6    , 4 , NULL              ,          ""      , showTemps2,       3,            _LCDML_TYPE_dynParam);
LCDML_addAdvanced (11 , LCDML_0_1_6    , 5 , NULL              ,          ""      , showTemps2,       4,            _LCDML_TYPE_dynParam);
LCDML_add         (12 , LCDML_0_1_6    , 6 , "Back"            , mFunc_back);

LCDML_add         (13, LCDML_0         , 2  , "Configure PID params"  , NULL);
LCDML_add         (14, LCDML_0_2       , 1  , "Setpoint"  , NULL);
LCDML_add         (15 , LCDML_0_2_1    , 1  , "", mFunc_back);
LCDML_addAdvanced (16 , LCDML_0_2_1    , 2  , NULL              ,          ""      , setReferenceTemp,       1,            _LCDML_TYPE_dynParam);
LCDML_addAdvanced (17 , LCDML_0_2_1    , 3  , NULL              ,          ""      , setReferenceTemp,       2,            _LCDML_TYPE_dynParam);
LCDML_addAdvanced (18 , LCDML_0_2_1    , 4  , NULL              ,          ""      , setReferenceTemp,       3,            _LCDML_TYPE_dynParam);
LCDML_addAdvanced (19 , LCDML_0_2_1    , 5  , NULL              ,          ""      , setReferenceTemp,       4,            _LCDML_TYPE_dynParam);
LCDML_add         (20 , LCDML_0_2_1    , 6  , "Show more"            , NULL);
LCDML_add         (21 , LCDML_0_2_1_6  , 1  , "", mFunc_back);
LCDML_addAdvanced (22 , LCDML_0_2_1_6  , 2  , NULL              ,          ""      , setReferenceTemp2,       1,            _LCDML_TYPE_dynParam);
LCDML_addAdvanced (23 , LCDML_0_2_1_6  , 3  , NULL              ,          ""      , setReferenceTemp2,       2,            _LCDML_TYPE_dynParam);
LCDML_addAdvanced (24 , LCDML_0_2_1_6  , 4 , NULL              ,          ""      , setReferenceTemp2,       3,            _LCDML_TYPE_dynParam);
LCDML_addAdvanced (25 , LCDML_0_2_1_6  , 5 , NULL              ,          ""      , setReferenceTemp2,       4,            _LCDML_TYPE_dynParam);
LCDML_add         (26 , LCDML_0_2_1_6  , 6 , "Back"            , mFunc_back);
LCDML_add         (27, LCDML_0_2       , 2  , "P"  , NULL);
LCDML_addAdvanced (28 , LCDML_0_2_2    , 1  , NULL              ,          ""      , setP,       1,            _LCDML_TYPE_dynParam);
LCDML_addAdvanced (29 , LCDML_0_2_2    , 2  , NULL              ,          ""      , setP,       2,            _LCDML_TYPE_dynParam);
LCDML_addAdvanced (30 , LCDML_0_2_2    , 3  , NULL              ,          ""      , setP,       3,            _LCDML_TYPE_dynParam);
LCDML_addAdvanced (31 , LCDML_0_2_2    , 4  , NULL              ,          ""      , setP,       4,            _LCDML_TYPE_dynParam);
LCDML_addAdvanced (32 , LCDML_0_2_2    , 5  , NULL              ,          ""      , setP,       5,            _LCDML_TYPE_dynParam);
LCDML_add         (33 , LCDML_0_2_2    , 6  , "Show more"            , NULL);
LCDML_addAdvanced (34 , LCDML_0_2_2_6  , 1  , NULL              ,          ""      , setP2,       1,            _LCDML_TYPE_dynParam);
LCDML_addAdvanced (35 , LCDML_0_2_2_6  , 2  , NULL              ,          ""      , setP2,       2,            _LCDML_TYPE_dynParam);
LCDML_addAdvanced (36 , LCDML_0_2_2_6  , 3  , NULL              ,          ""      , setP2,       3,            _LCDML_TYPE_dynParam);
LCDML_addAdvanced (37 , LCDML_0_2_2_6  , 4 , NULL              ,          ""      , setP2,       4,            _LCDML_TYPE_dynParam);
LCDML_addAdvanced (38 , LCDML_0_2_2_6  , 5 , NULL              ,          ""      , setP2,       5,            _LCDML_TYPE_dynParam);
LCDML_add         (39 , LCDML_0_2_2_6  , 6 , "Back"            , mFunc_back);
LCDML_add         (40, LCDML_0_2       , 3  , "I"  , NULL);
LCDML_addAdvanced (41 , LCDML_0_2_3    , 1  , NULL              ,          ""      , setI,       1,            _LCDML_TYPE_dynParam);
LCDML_addAdvanced (42 , LCDML_0_2_3    , 2  , NULL              ,          ""      , setI,       2,            _LCDML_TYPE_dynParam);
LCDML_addAdvanced (43 , LCDML_0_2_3    , 3  , NULL              ,          ""      , setI,       3,            _LCDML_TYPE_dynParam);
LCDML_addAdvanced (44 , LCDML_0_2_3    , 4  , NULL              ,          ""      , setI,       4,            _LCDML_TYPE_dynParam);
LCDML_addAdvanced (45 , LCDML_0_2_3    , 5  , NULL              ,          ""      , setI,       5,            _LCDML_TYPE_dynParam);
LCDML_add         (46 , LCDML_0_2_3    , 6  , "Show more"            , NULL);
LCDML_addAdvanced (47 , LCDML_0_2_3_6  , 1  , NULL              ,          ""      , setI2,       1,            _LCDML_TYPE_dynParam);
LCDML_addAdvanced (48 , LCDML_0_2_3_6  , 2  , NULL              ,          ""      , setI2,       2,            _LCDML_TYPE_dynParam);
LCDML_addAdvanced (49 , LCDML_0_2_3_6  , 3  , NULL              ,          ""      , setI2,       3,            _LCDML_TYPE_dynParam);
LCDML_addAdvanced (50 , LCDML_0_2_3_6  , 4 , NULL              ,          ""      , setI2,       4,            _LCDML_TYPE_dynParam);
LCDML_addAdvanced (51 , LCDML_0_2_3_6  , 5 , NULL              ,          ""      , setI2,       5,            _LCDML_TYPE_dynParam);
LCDML_add         (52 , LCDML_0_2_3_6  , 6 , "Back"            , mFunc_back);
LCDML_add         (53, LCDML_0_2       , 4  , "D"  , NULL);
LCDML_addAdvanced (54 , LCDML_0_2_4    , 1  , NULL              ,          ""      , setD,       1,            _LCDML_TYPE_dynParam);
LCDML_addAdvanced (55 , LCDML_0_2_4    , 2  , NULL              ,          ""      , setD,       2,            _LCDML_TYPE_dynParam);
LCDML_addAdvanced (56 , LCDML_0_2_4    , 3  , NULL              ,          ""      , setD,       3,            _LCDML_TYPE_dynParam);
LCDML_addAdvanced (57 , LCDML_0_2_4    , 4  , NULL              ,          ""      , setD,       4,            _LCDML_TYPE_dynParam);
LCDML_addAdvanced (58 , LCDML_0_2_4    , 5  , NULL              ,          ""      , setD,       5,            _LCDML_TYPE_dynParam);
LCDML_add         (59 , LCDML_0_2_4    , 6  , "Show more"            , NULL);
LCDML_addAdvanced (60 , LCDML_0_2_4_6  , 1  , NULL              ,          ""      , setD2,       1,            _LCDML_TYPE_dynParam);
LCDML_addAdvanced (61 , LCDML_0_2_4_6  , 2  , NULL              ,          ""      , setD2,       2,            _LCDML_TYPE_dynParam);
LCDML_addAdvanced (62 , LCDML_0_2_4_6  , 3  , NULL              ,          ""      , setD2,       3,            _LCDML_TYPE_dynParam);
LCDML_addAdvanced (63 , LCDML_0_2_4_6  , 4 , NULL              ,          ""      , setD2,       4,            _LCDML_TYPE_dynParam);
LCDML_addAdvanced (64 , LCDML_0_2_4_6  , 5 , NULL              ,          ""      , setD2,       5,            _LCDML_TYPE_dynParam);
LCDML_add         (65 , LCDML_0_2_4_6  , 6 , "Back"            , mFunc_back);

LCDML_add         (66 , LCDML_0_2      , 5  , "Load PID params"  , NULL);
LCDML_add         (67 , LCDML_0_2_5    , 1  , "Confirm"  , mFunc_load_pid_values);
LCDML_add         (68 , LCDML_0_2_5    , 2  , "Back"            , mFunc_back);
LCDML_add         (69 , LCDML_0_2      , 6  , "Save PID params"  , NULL);
LCDML_add         (70 , LCDML_0_2_6    , 1  , "Confirm"  , mFunc_save_pid_values);
LCDML_add         (71 , LCDML_0_2_6    , 2  , "Back"            , mFunc_back);


LCDML_add         (72 , LCDML_0        , 3  , "Plot Graph"  , NULL);
LCDML_add         (73,  LCDML_0_3      , 1  , NULL              ,          "Channel 1"      , realtimeGraph,       1,            _LCDML_TYPE_default);
LCDML_add         (74 , LCDML_0_3      , 2  , NULL              ,          "Channel 2"      , realtimeGraph,       2,            _LCDML_TYPE_default);
LCDML_add         (75 , LCDML_0_3      , 3  , NULL              ,          "Channel 3"      , realtimeGraph,       3,            _LCDML_TYPE_default);
LCDML_add         (76 , LCDML_0_3      , 4  , NULL              ,          "Channel 4"      , realtimeGraph,       4,            _LCDML_TYPE_default);
LCDML_addAdvanced (77 , LCDML_0_3      , 5  , "Show more"            , NULL);
LCDML_addAdvanced (78 , LCDML_0_3_5    , 1  , NULL              ,          "Channel 5"      , realtimeGraph,       5,            _LCDML_TYPE_default);
LCDML_addAdvanced (79 , LCDML_0_3_5    , 2  , NULL              ,          "Channel 6"      , realtimeGraph,       6,            _LCDML_TYPE_default);
LCDML_addAdvanced (80 , LCDML_0_3_5    , 3  , NULL              ,          "Channel 7"      , realtimeGraph,       7,            _LCDML_TYPE_default);
LCDML_add         (81 , LCDML_0_3_5    , 4  , NULL              ,          "Channel 8"      , realtimeGraph,       8,            _LCDML_TYPE_default);
LCDML_addAdvanced (82 , LCDML_0_3_5    , 5  , "Back"  , mFunc_back);
LCDML_addAdvanced (83 , LCDML_0_3      , 6  , "Back"  , mFunc_back);


LCDML_addAdvanced (84 , LCDML_0        , 4  , "Show PWM output"  , NULL);
LCDML_add         (85,  LCDML_0_4      , 1  , NULL              ,          ""      , show_PWM_output,       1,            _LCDML_TYPE_default);
LCDML_add         (86 , LCDML_0_4      , 2  , NULL              ,          ""      , show_PWM_output,       2,            _LCDML_TYPE_default);
LCDML_add         (87 , LCDML_0_4      , 3  , NULL              ,          ""      , show_PWM_output,       3,            _LCDML_TYPE_default);
LCDML_add         (88 , LCDML_0_4      , 4  , NULL              ,          ""      , show_PWM_output,       4,            _LCDML_TYPE_default);
LCDML_addAdvanced (82 , LCDML_0_4      , 5  , "Back"  , mFunc_back);


LCDML_addAdvanced (89 , LCDML_0        , 5  , "EMERGENCY STOP"  , NULL);
LCDML_add         (90 , LCDML_0_5      , 1  , "Confirm STOP"  , mFunc_emergencyStop);
LCDML_add         (91 , LCDML_0_5      , 2  , "Restart operation"  , mFunc_restartOperation);
LCDML_add         (92 , LCDML_0_5      , 3  , "Back"  , mFunc_back);