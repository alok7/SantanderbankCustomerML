import numpy as np

def normalize(data):
    '''This function finds the mean and std deviation, subtracts the mean from data and then divides it by std deviation. This assumes that the data is a numpy array'''
    mean = np.average(data,axis=0)
    std_dev = np.std(data,axis=0)

    # Avoiding dividing by 0
    eps = 1e-4
    std_dev[(std_dev>-eps) & (std_dev<eps)] = 1.0
    
    data[:,:] = data - mean
    data[:,:] = data/std_dev
    stats = np.array([mean,std_dev])
    return stats;

def normalize_stats(data,stats):
    '''Given the mean and standard deviation, this function normalizes the data.'''
    data[:,:] = data - stats[0]
    data[:,:] = data/stats[1]
    return;

def row_to_input(row_list):
    "This function converts our row data structure to numpy arrays which are suitable for the regression function of scikit learn."
    output_stripped = [row[:-1] for row in row_list]
    # Deleting the fields row
    return np.array(output_stripped[1:])

def row_to_output(row_list):
    "This function converts our row data structure to numpy arrays which are suitable for the regression function of scikit learn."
    output_stripped = [row[-1] for row in row_list]
    # Deleting the fields row
    return np.array(output_stripped[1:])

def row_convert_float(row):
    new_row = []
    for element in row:
        new_row.append(float(element))
    return new_row                

def sigmoid_array(x):                                        
    return 1 / (1 + np.exp(-x))
