import numpy as np

#%% numpy, example 1
#list1 = [12, 5, 7, -4]
#arr1 = np.array([12, 5, 7, -4])
#for element in list1:
#    print(element)

#for element in arr1:
#    print(element)

#for i in range(0,len(list1),1):
#    print(list1[i])

#dimension = arr1.shape
#for i in range(0,dimension[0],1):
#    print(arr1[i])


#%% numpy, example 2
#list1 = [12, 'UTE HCMC', 7, -4]
#arr1 = np.array([12, 'UTE HCMC', 7, -4])
#print(list1)
#print(arr1)


#%% numpy, example 3: matrices (2D array)
#mat1 = np.array([[2, 7, 10], [4, -8, 12], [24, -28, 212], [34, -38, 312]])
#dimension = mat1.shape
#print(dimension)


#%% numpy, example 4: slicing
#arr1 = np.array([12, 5, -1, 22, 8, 17, -4])
#print(arr1[2:6:1]) # [start:stop:step]
#print(arr1[5:1:-1])
#print(arr1[0:5:2])
#print(arr1[4::1])
#print(arr1[3::-1])
#print(arr1[::])
#print(arr1[::-1])
#print(arr1[:])

#mat1 = np.array([[2, 7, 10], [4, -8, 12], [24, -28, 212], [34, -38, 312]])
#print(mat1[0,2])
#print(mat1[2,0])
#print(mat1[1,:])
#print(mat1[:, 2])


import matplotlib.pyplot as plt
#%% matplotlib, example 1
X = np.linspace(-5,5,100)
Y = np.sin(X)*np.cos(X)
#Y = 2*X**2 + 4*X  
plt.plot(X,Y, color = 'r', linewidth = 3)

#Y = np.sin(X)  
#Y = 2*X**3 + 4*X  
Y = np.cos(np.exp(X)) 
plt.plot(X,Y, color = 'g', linewidth = 3)

# Plot Ox, Oy axis: exercise

#plt.axis([-10, 10, -100, 100])
plt.xlabel("Ox")
plt.ylabel("Oy")
plt.title("Graphs of some functions")
plt.legend(['Quadratic func.', 'Cubic func.'])
plt.show()


import pandas as pd
#%% pandas, example 1
#data = pd.read_csv(r'D:\OneDrive\02_Work\At_UTE\GiangDay\AI\qSlides\2020-2021 Sem1\Data_Images\PhanBoLaoDongTheoNganhNghe.csv')

#data.info()
#print("=======================================")
#print(data.head(4))
#print("=======================================")

## Refer to data parts
##print(data.iloc[[1,2,6], [2,3,4]])
##print(data.iloc[:, [2,3,4]])
#print(data.iloc[[1,2,6], :])

## Plot data
#fig = plt.figure()
#ax = fig.add_subplot(111)

#X = range(1,14,1)
#Y = data.iloc[1, :]
#plt.plot(X,Y[1::])

#Y = data.iloc[9, :]
#plt.plot(X,Y[1::])

#plt.legend(["Nong nghiep", "Dich vu an uong"])
#ax.set_xticks(range(1,14,2))
#ax.set_xticklabels(data.columns[1::2])
#plt.show()


#%% OOP, example 1
#class Pet: 
#    species = "unknown"
#    name = "unknown"

#    def __init__(self, species, name):
#        self.species = species
#        self.name = name

#    def get_name(self):
#        print("My name is", self.name)

#class Dog(Pet):
#    breed = ""

#    def __init__(self, name, breed):
#        Pet.__init__(self,'dog',name)
#        self.breed = breed

#    def bark(self, arr1):
#        print(arr1)


#pet1 = Pet('cat', 'Lucky')
#pet1.get_name()

#pet2 = Dog('Milu', 'Cho ta')
#pet2.get_name()
#pet2.bark([12, 3, 7, -9])







