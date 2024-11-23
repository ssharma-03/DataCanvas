import matplotlib.pyplot as plt
import os 

def create_bar_chart(data, x_column, y_column, output_file):
    plt.figure(figsize=(8, 6))
    plt.bar(data[x_column], data[y_column], color='skyblue')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title('Bar Chart')
    plt.savefig(output_file)
