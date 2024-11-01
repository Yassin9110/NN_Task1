from tkinter import ttk
import pandas as pd
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import seaborn as sns
from Preprocessing import Data_Preprocessing
from Algorithms import Adaline, SingleLayerPerceptron
from tkinter import messagebox
from matplotlib.colors import ListedColormap



class GUI:
    def __init__(self, master):
        self.master = master
        master.title("Bird Classification")

        self.feature1_label = tk.Label(master, text="Feature 1:")
        self.feature1_label.grid(row=0, column=0, padx=10, pady=5)

        feature_options = ["gender", "body_mass","beak_length",	"beak_depth", "fin_length"]
        self.feature1_combobox = ttk.Combobox(master, values=feature_options)
        self.feature1_combobox.set("gender")
        
        self.feature2_label = tk.Label(master, text="Feature 2:")
        self.feature2_combobox = ttk.Combobox(master, values=feature_options)
        self.feature2_combobox.set("body_mass")
        
        Classes_options = ["A", "B", "C"]
        self.class1_label = tk.Label(master, text="Class 1:")
        self.class1_entry = ttk.Combobox(master, values=Classes_options)
        self.class1_entry.set("A")
        self.class2_label = tk.Label(master, text="Class 2:")
        self.class2_entry = ttk.Combobox(master, values=Classes_options)
        self.class2_entry.set("B")

        self.learning_rate_label = tk.Label(master, text="Learning Rate:")
        self.learning_rate_entry = tk.Entry(master)
        self.learning_rate_entry.insert(0, "0.01")
        self.epochs_label = tk.Label(master, text="Epochs:")
        self.epochs_entry = tk.Entry(master)
        self.epochs_entry.insert(0, "30")
        self.mse_threshold_label = tk.Label(master, text="MSE Threshold:")
        self.mse_threshold_entry = tk.Entry(master)
        self.mse_threshold_entry.insert(0, "0.5")

        self.add_bias_var = tk.BooleanVar()
        self.add_bias_checkbox = tk.Checkbutton(master, text="Add Bias", variable=self.add_bias_var)

        self.algorithm_var = tk.StringVar(value="adaline")
        self.perceptron_radio = tk.Radiobutton(master, text="Perceptron", variable=self.algorithm_var, value="perceptron")
        self.adaline_radio = tk.Radiobutton(master, text="Adaline", variable=self.algorithm_var, value="adaline")



        self.train_button = tk.Button(master, text="Train", command=self.train_model)
        self.plot_button = tk.Button(master, text="Plot", command=self.plot_decision_boundary_and_confusion_matrix)

        
        self.feature1_label.grid(row=0, column=0)
        self.feature1_combobox.grid(row=0, column=1)
        self.feature2_label.grid(row=1, column=0)
        self.feature2_combobox.grid(row=1, column=1)
        self.class1_label.grid(row=2, column=0)
        self.class1_entry.grid(row=2, column=1)
        self.class2_label.grid(row=3, column=0)
        self.class2_entry.grid(row=3, column=1)
        self.learning_rate_label.grid(row=4, column=0)
        self.learning_rate_entry.grid(row=4, column=1)
        self.epochs_label.grid(row=5, column=0)
        self.epochs_entry.grid(row=5, column=1)
        self.mse_threshold_label.grid(row=6, column=0)
        self.mse_threshold_entry.grid(row=6, column=1)
        self.add_bias_checkbox.grid(row=7, column=0)
        self.perceptron_radio.grid(row=8, column=0)
        self.adaline_radio.grid(row=8, column=1)
        self.train_button.grid(row=9, column=0)
        self.plot_button.grid(row=9, column=1)

         # Output display area for metrics
        self.output_text = tk.Text(master, height=8, width=50)
        self.output_text.grid(row=8, column=4, columnspan=2, padx=12, pady=12)

    def display_results(self, text):
        self.output_text.insert(tk.END, text + "\n")
        self.output_text.see(tk.END) 

    def train_model(self):
        # Retrieve user input
        feature1 = self.feature1_combobox.get() 
        print(f"Feature 1 : {feature1}")
        feature2 = self.feature2_combobox.get()
        print(f"Feature 2 : {feature2}")
        if feature1 == feature2:
            messagebox.showerror("Error!", "The feature 1 & feature 2 can't be the same")
            return

        class1 = self.class1_entry.get()
        class2 = self.class2_entry.get()
        if class1 == class2:
            messagebox.showerror("Error!", "The class 1 & class 2 can't be the same")
            return

        learning_rate = float(self.learning_rate_entry.get())
        print(f"The lr = {learning_rate}")
        epochs = int(self.epochs_entry.get())
        print(f"The epochs = {epochs}")
        mse_threshold = float(self.mse_threshold_entry.get())
        print(f"The thresh = {mse_threshold}")
        add_bias = self.add_bias_var.get()
        
        algorithm = self.algorithm_var.get()

        # Preprocess the data
        preprocessor = Data_Preprocessing(feature1, feature2, class1, class2)
        preprocessor.preprocessing()
        
        X_train, X_test, y_train, y_test = preprocessor.splitdata()

        self.x_train = X_train
        self.x_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        # Train the selected algorithm
        if algorithm == "perceptron":
            model = SingleLayerPerceptron(learning_rate, epochs, add_bias)
        elif algorithm == "adaline":
            model = Adaline(learning_rate, epochs, mse_threshold, add_bias)
        else:
            print("Invalid algorithm selected.")
            return

        model.fit(X_train.to_numpy(), y_train.to_numpy())

        # Evaluate the model
        if algorithm == "perceptron":
            predictions = model.predict(X_test.to_numpy())
            accuracy = np.mean(predictions == y_test.to_numpy())
            self.predictions = predictions
            print(f'Accuracy using {algorithm}: {accuracy * 100:.2f}%')
            conf_matrix = model.confusion_matrix(y_test.to_numpy(), predictions)
        elif algorithm == "adaline":
            adaline_predictions = model.predict(X_test.to_numpy())
            self.predictions = adaline_predictions
            accuracy = np.mean(adaline_predictions == y_test.to_numpy())
            print(f'Accuracy using {algorithm}: {accuracy * 100:.2f}%')
            conf_matrix = model.confusion_matrix(y_test.to_numpy(), adaline_predictions)

        print("Confusion Matrix:")
        print(conf_matrix)

        self.display_results(f"Accuracy: {accuracy * 100:.2f}%")
        self.conf_matrix = conf_matrix
        

        

    def plot_decision_boundary_and_confusion_matrix(self):
        # Create a new window for the plots
        plot_window = tk.Toplevel(self.master)
        plot_window.title("Decision Boundary and Confusion Matrix")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        scatter = ax1.scatter(self.x_test.iloc[:, 0], self.x_test.iloc[:, 1], 
                            c=self.predictions, cmap=ListedColormap(['#FF0000', '#0000FF']), edgecolor='k', alpha=0.7)
        ax1.set_title("Test Data and Decision Boundary")
        ax1.set_xlabel("Feature 1")
        ax1.set_ylabel("Feature 2")

        x_min, x_max = self.x_test.iloc[:, 0].min(), self.x_test.iloc[:, 0].max()
        y_min, y_max = self.x_test.iloc[:, 1].min(), self.x_test.iloc[:, 1].max()
        ax1.plot([x_min, x_max], [y_min, y_max], color='black', linestyle='--', linewidth=1, label="Approx. Boundary")

        # Add legend to scatter plot
        legend1 = ax1.legend(*scatter.legend_elements(), title="Classes")
        ax1.add_artist(legend1)


        # Plot the confusion matrix
        sns.heatmap(self.conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax2)
        ax2.set_title("Confusion Matrix")
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Actual")

        # Display the plots in the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack()

if __name__ == "__main__":
    root = tk.Tk()
    gui = GUI(root)
    root.mainloop()