#Importing all required Libraries
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

#running the simulation a 'sims' times
sims = 10000000

#rolling dices a and b for 'sims' number of times
a = np.random.randint(1, 7, sims)
b = np.random.randint(1, 7, sims)

#finding the sum of 2 dices
x_sum = a + b

#plotting the histogram
plt.figure(figsize= (6, 4))
plt.hist(x_sum, bins=range(2,14), density= True)
#we win the game at 10 or more
plt.axvline(10, color = 'r')
plt.xlabel("Sum of two dice")
plt.ylabel("Frequency")
plt.title("Monte Carlo Dice Simulation")
plt.show()

#printing the probablity to win
print((x_sum >= 10).sum()/sims)
print("Done")