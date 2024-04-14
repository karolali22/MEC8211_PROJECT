import os
import re
import numpy as np

def extract_values(filename):
    with open(filename, 'r') as file:
        content = file.read()
        aoa_match = re.search(r'ALPHA = ([\d.-]+e[\d+-]+)', content)
        cd_match = re.search(r'CD = ([\d.-]+e[\d+-]+)', content)
        aoa = float(aoa_match.group(1)) if aoa_match else None
        cd = float(cd_match.group(1)) if cd_match else None
    return aoa, cd

def compute_order_of_convergence(f_coarse, f_medium, f_fine):
    try:
        return np.log(abs((f_coarse - f_medium) / (f_medium - f_fine))) / np.log(2)
    except ZeroDivisionError:
        return None

def compute_gci(p_hat, p_f, f_medium, f_fine):
    epsilon = abs(p_hat - p_f) / p_f
    if epsilon < 0.01:
        return 0
    elif epsilon <= 0.1:
        return 1.25 * abs((f_fine - f_medium) / (2**p_f - 1))
    else:
        p = min(max(0.5, p_hat), p_f)
        return 3 * abs((f_fine - f_medium) / (2**p - 1))

def gather_data_and_compute(folder_path):
    folders = ['../data/convergence_results/65x65', 
               '../data/convergence_results/129x129', 
               '../data/convergence_results/257x257']
    data = {}

    for folder in folders:
        dir_path = os.path.join(folder_path, folder)
        for filename in os.listdir(dir_path):
            full_path = os.path.join(dir_path, filename)
            aoa, cd = extract_values(full_path)
            if aoa and cd is not None:
                if aoa not in data:
                    data[aoa] = {}
                data[aoa][folder] = cd

    results = {}
    for aoa, values in data.items():
        if all(folder in values for folder in folders):
            p_hat = compute_order_of_convergence(values['../data/convergence_results/65x65'], 
                                                 values['../data/convergence_results/129x129'], 
                                                 values['../data/convergence_results/257x257'])
            if p_hat is not None:
                gci = compute_gci(p_hat, 1, values['../data/convergence_results/129x129'], 
                                            values['../data/convergence_results/257x257'])  # Assuming p_f = 1
                results[aoa] = {
                    'p_hat': p_hat, 'GCI': gci,
                    'CD_65': values['../data/convergence_results/65x65'],
                    'CD_129': values['../data/convergence_results/129x129'], 
                    'CD_257': values['../data/convergence_results/257x257']
                }
            else:
                results[aoa] = {'Error': 'Division by zero'}
        else:
            results[aoa] = {'Error': 'Not enough data'}

    return results

def main():
    folder_path = os.getcwd()
    results = gather_data_and_compute(folder_path)
    with open('../data/CD/num_uncertainty_CD.txt', 'w') as file:
        for aoa in sorted(results.keys(), key=float):
            result = results[aoa]
            file.write(f"AOA {aoa}: {result}\n")

if __name__ == '__main__':
    main()
