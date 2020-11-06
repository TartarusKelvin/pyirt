import numpy as np

def spearmen_brown(data,split="half"):
    first_half = []
    second_half = []
    if split == "half":
        first_half = np.array(data[:int(len(data)/2)])
        second_half = np.array(data[int(len(data)/2):])
    elif split == "odd":
        first_half = data[::2]
        second_half = data[1::2]
    else:
        raise ValueError("Unregonised split type, valid types are: half, odd")

    first_sums = first_half.sum(axis = 0) 
    second_sums = second_half.sum(axis = 0) 
    correlation = np.corrcoef(first_sums, second_sums)[0, 1]

    return 2*correlation/(1+correlation)


def cronbach_alpha (data):
    data = np.array(data)

    k = data.shape[1]

    covariances = np.corrcoef(data)
    v_bar = covariances.trace()/k
    np.fill_diagonal(covariances, None)
    c_bar = np.mean(covariances[~(np.isnan(covariances))])
    
    return k*c_bar/(v_bar + (k-1)*c_bar)

print(cronbach_alpha(
    [[10,20,30],
    [30,40,50],
    [50,60,70],
    [70,80,90],
    [90,100,110]]
))
