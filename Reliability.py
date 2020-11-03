import numpy as np

def SpearmenBrown(data,split="half"):
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


print(SpearmenBrown(
    [[0,1,3,2,4,6,7,1,0,2,0,3],
    [1,1,2,2,1,1,3,0,4,3,1,0],
    [2,0,3,4,1,0,4,1,6,2,0,2],
    [0,3,0,2,3,3,6,0,3,3,1,3],
    [1,2,1,1,4,5,2,1,0,2,2,4],]
))
