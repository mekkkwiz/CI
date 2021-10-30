import numpy as np
import matplotlib.pyplot as plt

def temp_low_fuzzification (temp):
    if (temp <= 10):
        return 1
    elif (10 <= temp <= 20):
        return (20 - temp)/10
    else:
        return 0

def temp_mid_fuzzification (temp):
    if (11 <= temp <= 20):
        return (temp - 11)/9
    elif (20 <= temp <= 28):
        return (28 - temp)/8
    else:
        return 0

def temp_high_fuzzification (temp):
    if (20 <= temp <= 28):
        return (temp - 20)/8
    elif (28 <= temp <= 38):
        return 1
    else:
        return 0

def hum_optimal_fuzzification (temp):
    if(10 <= temp <= 11):
        return 1
    if(11 <= temp <= 14):
        return (14-temp)/3
    else:
        return 0

def hum_humid_fuzzification (temp):
    if(12 <= temp <= 15):
        return (temp-12)/3
    if(15 <= temp <= 18):
        return 1
    else:
        return 0

def cs_low_defuzz (value):
    if (0 <= value <= 30):
        return 1
    elif (30 <= value <= 50):
        return (50-value)/20
    else:
        return 0

def cs_mid_defuzz (value):
    if (40 <= value <= 60):
        return (value - 40)/20
    elif (60 <= value <= 80):
        return (80-value)/20
    else:
        return 0     

def cs_fast_defuzz (value):
    if (70 <= value <= 90):
        return (value -70)/20
    elif (90 <= value <= 100):
        return 1
    else:
        return 0

def fs_low_defuzz (value):
    if (0 <= value <= 30):
        return 1
    elif (31 <= value <= 50):
        return (50-value)/20
    else:
        return 0

def fs_mid_defuzz (value):
    if (40 <= value <= 60):
        return (value - 40)/20
    elif (61 <= value <= 80):
        return (80-value)/20
    else:
        return 0     

def fs_fast_defuzz (value):
    if (70 <= value <= 90):
        return (value - 70)/20
    elif (91 <= value <= 100):
        return 1
    else:
        return 0


def mamdani_2in_2out (temp, hum):
    t_l_f = temp_low_fuzzification(temp)
    t_m_f = temp_mid_fuzzification(temp)
    t_h_f = temp_high_fuzzification(temp)

    h_o_f = hum_optimal_fuzzification(hum)
    h_h_f = hum_humid_fuzzification(hum)

    temp_normalized = (temp-10)/(38-10)
    hum_normalized = (hum-10)/(18-10)

    y1 = (temp_normalized+hum_normalized)
    print(y1/2)

    fs_l_idx1 = 0
    fs_l_idx2 = 0
    while (not((y1/2)-0.001 <= fs_l_idx1 <= (y1/2)+0.001)):
        fs_l_idx1 = fs_low_defuzz(fs_l_idx2)
        fs_l_idx2 += 0.001

    fs_m_idx1 = 0
    fs_m_idx2 = 0
    while (not((y1/2)-0.001 <= fs_m_idx1 <= (y1/2)+0.001)):
        fs_m_idx1 = fs_mid_defuzz(fs_m_idx2)
        fs_m_idx2 += 0.001

    fs_f_idx1 = 0
    fs_f_idx2 = 0
    while (not((y1/2)-0.001 <= fs_f_idx1 <= (y1/2)+0.001)):
        fs_f_idx1 = fs_fast_defuzz(fs_f_idx2)
        fs_f_idx2 += 0.001
    
    cs_l_idx1 = 0
    cs_l_idx2 = 0
    while (not((y1/2)-0.001 <= cs_l_idx1 <= (y1/2)+0.001)):
        cs_l_idx1 = cs_low_defuzz(cs_l_idx2)
        cs_l_idx2 += 0.001

    cs_m_idx1 = 0
    cs_m_idx2 = 0
    while (not((y1/2)-0.001 <= cs_m_idx1 <= (y1/2)+0.001)):
        cs_m_idx1 = cs_mid_defuzz(cs_m_idx2)
        cs_m_idx2 += 0.001

    cs_f_idx1 = 0
    cs_f_idx2 = 0
    while (not((y1/2)-0.001 <= cs_f_idx1 <= (y1/2)+0.001)):
        cs_f_idx1 = cs_fast_defuzz(cs_f_idx2)
        cs_f_idx2 += 0.001
    
    temp = None
    hum = None
    if(max(t_l_f,t_m_f,t_h_f) == t_l_f):
        temp = ["low", t_l_f]
    elif(max(t_l_f,t_m_f,t_h_f) == t_m_f):
        temp = ["mid", t_m_f]
    elif(max(t_l_f,t_m_f,t_h_f) == t_h_f):
        temp = ["high", t_h_f]
    
    if(max(h_o_f,h_h_f) == h_o_f):
        hum = ["optimal", h_o_f]
    elif(max(h_o_f,h_h_f) == h_h_f):
        hum = ["humid", h_h_f]

    # rule checking
    # print(temp, hum)

    if(temp[0] == "low" and hum[0] == "optimal"):
        print("output fs%:", fs_l_idx2)
        print("output cs%:", cs_l_idx2)
        return fs_l_idx2,cs_l_idx2
    elif(temp[0] == "mid" and hum[0] == "optimal"):
        print("output fs%:", fs_m_idx2)
        print("output cs%:", cs_l_idx2)
        return fs_m_idx2, cs_l_idx2
    elif(temp[0] == "high" and hum[0] == "optimal"):
        print("output fs%:", fs_f_idx2)
        print("output cs%:", cs_f_idx2)
        return fs_f_idx2, cs_f_idx2
    elif(temp[0] == "low" and hum[0] == "humid"):
        print("output fs%:", fs_l_idx2)
        print("output cs%:", cs_l_idx2)
        return fs_l_idx2, cs_l_idx2
    elif(temp[0] == "mid" and hum[0] == "humid"):
        print("output fs%:", fs_m_idx2)
        print("output cs%:", cs_l_idx2)
        return fs_m_idx2, cs_l_idx2
    elif(temp[0] == "high" and hum[0] == "humid"):
        print("output fs%:", fs_f_idx2)
        print("output cs%:", cs_f_idx2)
        return fs_f_idx2, cs_f_idx2
    else:
        print("None")
        return 0,0


res1 = []
res2 = []

for i in range(1,80):
    print("round:", i)
    print(11 + i%27, 10 + i%9)
    x1,x2 = mamdani_2in_2out(11 + i%27, 10 + i%8)
    res1.append(x1)
    res2.append(x2)


fig, (ax1, ax3) = plt.subplots(1, 2)
ax1.set_title('input')
ax3.set_title('output')

ax1.plot([i for i in range(1,80)],[11 + i%27 for i in range(1,80)], label = "temp")
ax1.plot([i for i in range(1,80)],[10 + i%8 for i in range(1,80)], label = "humidity")

print("------------")
ax3.plot([i for i in range(1,80)],res1, label ='fan(%)')
ax3.plot([i for i in range(1,80)],res2, label ='compressor(%)')

ax3.legend()
ax1.legend()
plt.show()