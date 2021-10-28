def temp_low_fuzzification (temp):
    if (temp <= 10):
        return 1
    elif (11 <= temp <= 20):
        return (20 - 10)/3
    else:
        return 0

def temp_mid_fuzzification (temp):
    if (11 <= temp <= 20):
        return (temp - 11)/3
    elif (21 <= temp <= 28):
        return (28 - temp)/3

def temp_high_fuzzification (temp):
    if (20 <= temp <= 28):
        return (temp - 22)/3
    elif (29 <= temp <= 38):
        return 1
    else:
        return 0

def hum_optimal_fuzzification (temp):
    if(10 <= temp <= 11):
        return 1
    if(12 <= temp <= 14):
        return (14-temp)/3
    else:
        return 0

def hum_humid_fuzzification (temp):
    if(12 <= temp <= 15):
        return (temp-12)/3
    if(16 <= temp <= 18):
        return 1
    else:
        return 0

def cs_low_fuzzification (value):
    if (0 <= value <= 30):
        return 1
    elif (31 <= value <= 50):
        return (50-value)/20
    else:
        return 0

def cs_low_fuzzification (value):
    if (0 <= value <= 30):
        return 1
    elif (31 <= value <= 50):
        return (50-value)/20
    else:
        return 0

def cs_mid_fuzzification (value):
    if (40 <= value <= 60):
        return (value - 40)/20
    elif (61 <= value <= 80):
        return (80-value)/20
    else:
        return 0     

def cs_fast_fuzzification (value):
    if (70 <= value <= 90):
        return (value -70)/20
    elif (91 <= value <= 100):
        return 1
    else:
        return 0

def fs_low_fuzzification (value):
    if (0 <= value <= 30):
        return 1
    elif (31 <= value <= 50):
        return (50-value)/20
    else:
        return 0

def fs_mid_fuzzification (value):
    if (40 <= value <= 60):
        return (value - 40)/20
    elif (61 <= value <= 80):
        return (80-value)/20
    else:
        return 0     

def fs_fast_fuzzification (value):
    if (70 <= value <= 90):
        return (value - 70)/20
    elif (91 <= value <= 100):
        return 1
    else:
        return 0

def mamdani_2in_2out (temp,hum):
    t_l_f = temp_low_fuzzification(temp)
    t_m_f = temp_mid_fuzzification(temp)
    t_h_f = temp_high_fuzzification(temp)

    h_o_f = hum_optimal_fuzzification(hum)
    h_h_f = hum_humid_fuzzification(hum)

    print(t_l_f,t_m_f,t_h_f)
    print(h_o_f,h_h_f)

mamdani_2in_2out(25,15)