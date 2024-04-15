import matplotlib.pyplot as plt

file_path = '../data/CD/num_uncertainty_CD.txt'

aoa_data = []
gci_data = []

with open(file_path, 'r') as file:
    for line in file:
        split_index = line.find(':')
        aoa_str = line[:split_index].strip()
        data_str = line[split_index + 1:].strip()
        
        aoa = float(aoa_str.split(' ')[1])
        data_dict = eval(data_str)

        aoa_data.append(aoa)
        gci_data.append(data_dict['GCI'])


plt.figure(figsize=(6, 5))
plt.scatter(aoa_data, gci_data, marker='x')
plt.xlabel(r'Angle of Attack $\alpha$ [°]', fontsize=12)
plt.ylabel(r'GCI', fontsize=12)
plt.ylim(-0.5, 2.5)
plt.xlim(-5, 20)
plt.savefig('../results/GCI_CD.png', dpi=600)
plt.show()
