# Plot-Program
Python program for data visualization and evaluation. The working title of this program is: 

<div align="center">
<h3>
Graph interactive editor for smooth visualization GIES v.1.0
</h3>
</div>

The program is intended to be used as a tool for a quick data visualization tool to view experimental data with one or two rows of data.

# Explanation of the code

The code is structured into several class which inherit properties of the Customtkinter package, which can be found here:

**https://github.com/TomSchimansky/CustomTkinter/tree/master**

## class App(customtkinter.CTk):
This class inherits from the CTk class of customtkinter which has several build in tools to generate buttons, switches etc. on an overlay based on the tkinter package.

### initialize_ui():
- Initialize the user interface
- create all buttons, switches, entries and the dropdown menu

### Functions for creating the different widgets:

#### create_label(width, row, column):
- optional: text=None, anchor='e', sticky=None, textvariable=None, padx=None, pady=None, columnspan=1
- create a CTkLabel and pack it on the grid

#### create_entry(width, row, column):
- optional: text=None, columnspan=1, padx=10, pady=5, placeholder_text=None
- create a CTkEntry and pack it on the grid
- optional "text" parameter: create label to the left of the entry

#### create_button(text, command, width, row, column):
- optional: columnspan=1, padx=10, pady=5, sticky=None
- create a CTkSwitch and pack it on the grid 

#### create_switch(text, command, width, row, column):
- optional: columnspan=1, padx=10, pady=5
- create a CTkSwitch and pack it on the grid 

#### create_combobox(values, width, column, row,):
- optional: state='readonly', command=None, text=None, columnspan=1, padx=10, pady=5
- create a CTkCombobox and pack it on the grid
- optional "text" parameter: create label to the left of the combobox

#### create_slider( from_, to, width, row, column):
- optional: text=None, init_value=None, command=None, columnspan=1, number_of_steps=1000, padx = 10, pady=5
- create a CTkSlider and pack it on the grid
- optional "text" parameter: create label to the left of the slider
- optional "init_value" parameter: set initial slider value

## initialize_plot():
- initialize a new plot window, create a figure in matplotlib
- set colormap
- set xlabel and ylabel 
- create the canvas to display the plot 
- set a ymax value to track the maximum y value for all plots (if multiplot is used)
- call plot()

## plot():
- self.initialize_plot_has_been_called: ensure that initialize_plot is at least called in a session before a plot is created
- obtain the file path from the list
- determine the file decimal (either comma or dot)
- check if the data is a 1D array, then create a dummy array to represent the x-values
- check if normalize slider is active
- check if uselabels slider is active
- check if use_fit slider is active
- call update_plot()

## update_plot():
This function is called when there is no new data to be plotted and just plot adjustments are made
- normalize data (if normalize slider active)
- update limits 
- the canvas has to be redrawn

## change_plot(replot, direction):
This function is invoked by the << and >> buttons, to plot the next/previous plot in the file list 
- replot(Boolean): True - new plot is initialized, False - add plot to current figure
- direction(integer): 1 - next file, -1 - previous file

## fit_plot(function, initial_params):
Create a fit of the current self.data array and returns the fitted array
- function: exponential, gauss, sqrt, linear, quadratic
- initial_params: initial parameter, given by user input

## read_file_list():
Invoked when "Load Folder" is pressed
- reads from the current folder all files with extensions: csv, dat, txt 
- configures the optionsmenu to display all file names

## use_limits():
Invoked when "Use limits" switch is flipped
- Does nothing when no plot is active
- creates the labels of xlimits and ylimits 
- creates sliders for the left and right limits for x and y
- if Switch is inactive: remove everything from the grid
- call update_plot() 

## use_labels():
Invoked when "Use labels" switch is flipped
- creates entries for xlabel, ylabel and a plot legend
- if Switch is inactive: remove everything from the grid

## multiplot():
Invoked when "Multiplot" switch is flipped
- replaces the "Plot graph" button with two buttons for multiplot purposes
- plot2_button: same button as plot_button but different size (renamed as "reset")
- addplot_button: call plot() instead of initialize_plot()

## update_limits():
Invoked every time the slider value is changed for the limits. Reconfigure the x- and y-limits

## normalize_setup():
Invoked when "normalize" swtich is flipped
- reinitialize the plot (destroys all multiplots, because only the data of the current graph is stored)
- update the limits to work properly with "use limits" switch 
- reinitialize the plot again

## normalize():
Do the normalization of the data
- maximum: normalize such that the maximum value of the data is one
- area: normalize such that the total area of the plot is one

## skip_rows(file_path):
Determine the number of rows that have to be skipped in order to read the file. The purpose is to delete header lines from files. The function checks whether or not the first character of the first line is a number. If not, move to the next line and increase the counter for skip_rows (return value).

## save_figure():
Is called when the "save figure" button is pressed. Opens a file save dialog. The file format is not specified (user input).

## open_file():
Is called by plot(). It checks whether a document contains more dots or commas to determine the decimal separator. 

## open_toplevel():
Opens a new toplevel_window of the different classes

## on_closing():
Stop the script when the window is closed.