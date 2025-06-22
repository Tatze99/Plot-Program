# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 14:28:26 2022

@author: Martin

to do: add custom textbox, ax-lines
"""
import os
from CTkRangeSlider import *
import customtkinter
from CTkTable import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.colors import ListedColormap
import scipy.ndimage
# from matplotlib.figure import Figure
import ctypes
import matplotlib
from cycler import cycler
from scipy.optimize import curve_fit as cf
from scipy.interpolate import interp1d
from inspect import signature # get number of arguments of a function
from PIL import Image, ImageTk
from natsort import natsorted
import io # used for latex display buffer
import logging

myappid = 'mycompany.myproduct.subproduct.version' # arbitrary string
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

tooltips_enabled = True

def toggle_bool(event=None):
    global tooltips_enabled
    tooltips_enabled = not tooltips_enabled

version_number = "24/08"
Standard_path = os.path.dirname(os.path.abspath(__file__))
plt.style.use('default')
# plt.style.use(os.path.join(Standard_path, "dark_mode.mplstyle"))
matplotlib.rc('font', family='serif')
matplotlib.rc('font', serif='Times New Roman')
file_type_names = ('.csv', '.dat', '.txt', '.png', '.jpg', '.jpeg', '.spec', '.JPG', '.bmp', '.webp', '.tif', '.tiff', '.PNG', '.pgm', '.pbm', '.lvm')
image_type_names = ('png','.jpg', '.jpeg', '.JPG', '.bmp', '.webp', '.tif', '.tiff', '.PNG', '.pgm', '.pbm')
sequential_colormaps = ['magma','hot','viridis', 'plasma', 'inferno', 'cividis', 'gray', 'bone', 'afmhot', 'copper','Purples', 'Blues', 'Greens', 'Oranges', 'Reds','twilight', 'hsv', 'rainbow', 'jet', 'turbo', 'gnuplot', 'brg']

ctypes.windll.shcore.SetProcessDpiAwareness(1)

filelist = natsorted([fname for fname in os.listdir(Standard_path) if fname.endswith(file_type_names)])

def exponential(x, a, b, c): 
    return a * np.exp(b * x) + c

def lorentz(x, a, w0, gamma, d):
    return a**2 / ((x**2 - w0**2)**2 + gamma**2*w0**2) + d

def gauss(x, a, b, c, d):
    return a * np.exp(-(x - b)**2 / (2 * c**2)) + d

def gauss3(x, a, b, c, d):
    return a * np.exp(-abs(x - b)**3 / (2 * c**2)) + d

def sqrt(x, a, b, c):
    return a * np.sqrt(b * x) + c

def linear(x, a, b):
    return np.polyval([a,b], x)

def quadratic(x, a, b, c):
    return np.polyval([a,b,c], x)

def logarithm(x, a, b, c):
    return a * np.log(b * x) + c

def hyperbola(x, a, b, c):
    return a / (x-b) + c

def moving_average(x, window_size):
    """
    Smooths the given array x using a centered moving average.

    Parameters:
        x (list or numpy array): The input array to be smoothed.
        window_size (int): The size of the centered moving average window.

    Returns:
        smoothed_array (numpy array): The smoothed array.
    """
    # Ensure the window_size is even
    if window_size % 2 == 0:
        half_window = window_size // 2
    else:
        half_window = (window_size - 1) // 2

    if window_size <= 1:
        return x

    half_window = window_size // 2
    cumsum = np.cumsum(x)

    # Calculate the sum of elements for each centered window
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    centered_sums = cumsum[window_size - 1:-1]

    # Divide each sum by the window size to get the centered moving average
    smoothed_array = centered_sums / window_size

    # Pad the beginning and end of the smoothed array with the first and last values of x
    first_value = np.repeat(x[0], half_window)
    last_value = np.repeat(x[-1], half_window)
    smoothed_array = np.concatenate((first_value, smoothed_array, last_value))

    return smoothed_array

def calc_log_value(x, x_l, x_r):
    if x   <= 1e-6*x_r: x   = 1e-6*x_r
    if x_l <= 1e-6*x_r: x_l = 1e-6*x_r
    return np.exp(np.log(x_l) + np.log(x_r/x_l) * (x-x_l)/(x_r-x_l))

def get_border_points(x0, y0, angle, array_shape):
    theta = np.radians(angle)
    xdim, ydim = array_shape
    intersections = []

    if theta == 0:
        intersections = [(0, y0), (xdim-1, y0)]

    elif (angle == 90 or angle == -90):
        intersections = [(x0, 0), (x0, ydim-1)]

    else: 
        # Intersection with the left border (x = 0)
        t = -x0 / np.cos(theta)
        y = y0 + t * np.sin(theta)
        if 0 <= y < ydim:
            intersections.append((0, round(y)))
        
        # Intersection with the right border (x = xdim-1)
        t = (xdim-1 - x0) / np.cos(theta)
        y = y0 + t * np.sin(theta)
        if 0 <= y < ydim:
            intersections.append((xdim-1, round(y)))
        
        # Intersection with the top border (y = 0)
        t = -y0 / np.sin(theta)
        x = x0 + t * np.cos(theta)
        if 0 <= x < xdim:
            intersections.append((round(x), 0))
        
        # Intersection with the bottom border (y = ydim-1)
        t = (ydim-1 - y0) / np.sin(theta)
        x = x0 + t * np.cos(theta)
        if 0 <= x < xdim:
            intersections.append((round(x), ydim-1))
    
    # Assuming there are exactly two intersection points for a line intersecting the border
    if len(intersections) != 2:
        raise ValueError("The line does not intersect the border exactly twice.")
    
    return intersections

def find_fwhm(xdata, ydata):
    # Step 1: Find the maximum value and its index
    max_y = np.max(ydata)
    max_index = np.argmax(ydata)
    
    # Step 2: Calculate the half maximum
    half_max = max_y / 2.0
    
    # Step 3: Find the index where ydata first crosses the half maximum on the left side of the peak
    left_index = np.where(ydata[:max_index] <= half_max)[0]
    if len(left_index) > 0:
        left_index = left_index[-1]
    else:
        left_index = 0

    # Step 4: Find the index where ydata first crosses the half maximum on the right side of the peak
    right_index = np.where(ydata[max_index:] <= half_max)[0]
    if len(right_index) > 0:
        right_index = right_index[0] + max_index
    else:
        right_index = len(ydata) - 1

    # Step 5: Interpolation to find the exact position of half maximum on both sides
    x_left = xdata[left_index] + (xdata[left_index + 1] - xdata[left_index]) * \
             (half_max - ydata[left_index]) / (ydata[left_index + 1] - ydata[left_index])

    x_right = xdata[right_index - 1] + (xdata[right_index] - xdata[right_index - 1]) * \
              (half_max - ydata[right_index - 1]) / (ydata[right_index] - ydata[right_index - 1])

    # Step 6: Calculate the FWHM
    fwhm = x_right - x_left
    
    return x_left, x_right, half_max, fwhm

def flatten(data):
    for item in data:
        if isinstance(item, tuple):
            yield from flatten(item)
        else:
            yield item

def determine_delimiter_and_column_count(file_path, skip_row):
    with open(file_path, 'r') as file:
        # Skip lines until the desired line
        for _ in range(skip_row):
            file.readline()
        # Read the first line
        target_line = file.readline().strip()
        
        # Potential delimiters
        delimiters = [';', '\t',' ']
        if '.' in target_line:
            delimiters.append(',')
        delimiter = None
        max_columns = 0
        
        # Determine the delimiter by checking which one results in the most columns
        for d in delimiters:
            columns = target_line.split(d)
            if len(columns) > max_columns:
                max_columns = len(columns)
                if d == ' ': # handle any amount of white space
                    delimiter = None
                else:
                    delimiter = d
        

        return delimiter, max_columns

def format_number(val):
    # Format with 5 significant digits, using scientific notation if the value is very small or large
    if abs(val) < 0.001 or abs(val) >= 1e5:  # Threshold for switching to scientific notation
        return f"{val:.4e}"  # Scientific notation with 5 significant digits
    else:
        # if val < 0: fill_zeros += 1
        return f"{val:.4f}"  # Standard float with 5 significant digits
    
class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        customtkinter.set_appearance_mode("dark")
        customtkinter.set_default_color_theme("dark-blue")
        self.title("Graph interactive editor for smooth visualization GIES v."+version_number)
        self.geometry("1280x720")
        self.replot = False
        self.initialize_variables()
        self.initialize_ui_images()
        self.initialize_ui()
        self.initialize_plot_has_been_called = False
        self.folder_path.insert(0, Standard_path)
        self.toplevel_window = {'Plot Settings': None,
                                'Legend Settings': None}
        # hide the multiplot button
        self.addplot_button.grid_remove() 
        self.reset_button.grid_remove()
        
    def initialize_ui_images(self):
        self.img_settings = customtkinter.CTkImage(dark_image=Image.open(os.path.join(Standard_path,"ui_images","options.png")), size=(15, 15))
        self.img_save = customtkinter.CTkImage(dark_image=Image.open(os.path.join(Standard_path,"ui_images","save_white.png")), size=(15, 15))
        self.img_folder = customtkinter.CTkImage(dark_image=Image.open(os.path.join(Standard_path,"ui_images","folder.png")), size=(15, 15))
        self.img_reset = customtkinter.CTkImage(dark_image=Image.open(os.path.join(Standard_path,"ui_images","reset.png")), size=(15, 15))
        self.img_next = customtkinter.CTkImage(dark_image=Image.open(os.path.join(Standard_path,"ui_images","next_arrow.png")), size=(15, 15))
        self.img_previous = customtkinter.CTkImage(dark_image=Image.open(os.path.join(Standard_path,"ui_images","previous_arrow.png")), size=(15, 15))
        
    def initialize_variables(self):
        self.ax1 = None
        self.ax2 = None
        self.plot_counter = 1
        self.plot_index = 0
        self.color = "#212121" # toolbar
        self.text_color = "white"
        self.reset_plots = True
        self.legend_type = {'loc': 'best'}
        self.legend_name = slice(None, None)
        self.canvas_ratio = None #None - choose ratio automatically
        self.canvas_width = 17 #cm
        self.canvas_height= 10
        self.label_settings = "smart"
        self.ticks_settings = "smart"
        self.multiplot_row = 9
        self.legend_font_size = 10
        self.initialize_plot_has_been_called = False

        # line plot settings
        self.linestyle='-'
        self.markers=''
        self.cmap="tab10"
        self.linewidth=1
        self.alpha = 1
        self.moving_average = 1
        self.use_grid_lines = customtkinter.BooleanVar(value=False)
        self.use_minor_ticks = customtkinter.BooleanVar(value=False)
        self.grid_ticks = 'major'
        self.grid_axis = 'both'
        self.cmap_length = 10
        self.single_color = 'tab:blue'
        self.sub_plot_counter = 1
        self.sub_plot_value = 1
        self.first_ax2 = True
        self.draw_FWHM_line = customtkinter.BooleanVar(value=False)
        self.normalize_function = 'Maximum'
        self.normalize_value = 1
        
        # image plot settings
        self.cmap_imshow="magma"
        self.interpolation = None
        self.enhance_value = 1 
        self.pixel_size = 7.4e-3 # mm
        self.label_dist = 1
        self.aspect = "equal"
        self.plot_type = "errorbar"
        self.clim = (0,1)

        # fit settings
        self.use_fit = 0
        self.function = gauss
        self.params = [1,1,1,1]
        self.fitted_params = [0,0,0,0]

        # Boolean variables
        self.image_plot = True
        self.image_plot_prev = True
        self.save_plain_image = customtkinter.BooleanVar(value=False)
        self.display_fit_params_in_plot = customtkinter.BooleanVar(value=True)
        self.use_scalebar = customtkinter.BooleanVar(value=False)
        self.use_colorbar = customtkinter.BooleanVar(value=False)
        self.convert_pixels = customtkinter.BooleanVar(value=False)
        self.convert_rgb_to_gray = customtkinter.BooleanVar(value=False)
        self.update_labels_multiplot = customtkinter.BooleanVar(value=True)
        self.show_title_entry = customtkinter.BooleanVar(value=True)
        self.rows = 1
        self.cols = 1
        self.mouse_clicked_on_canvas = False
        
    # user interface, gets called when the program starts 
    def initialize_ui(self):

        self.sidebar_frame = customtkinter.CTkFrame(self, width=200, height=600, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=999, sticky="nesw")
        self.sidebar_frame.rowconfigure(15, weight=1)
        self.rowconfigure(11,weight=1)

        App.create_label(self.sidebar_frame, text="GIES v."+version_number, font=customtkinter.CTkFont(size=20, weight="bold"),row=0, column=0, padx=20, pady=(10,15), sticky=None)
        
        self.tabview = customtkinter.CTkTabview(self, width=250, command=lambda: self.toggle_toolbar(self.tabview.get()))
        self.tabview.grid(row=3, column=1, padx=(10, 10), pady=(20, 0), columnspan=2, sticky="nsew", rowspan=10)
        self.tabview.add("Show Plots")
        self.tabview.add("Data Table")
        self.tabview.tab("Show Plots").columnconfigure(0, weight=1)
        self.tabview.tab("Show Plots").rowconfigure(0, weight=1)

        #buttons
        frame = self.sidebar_frame
        
        self.plot_button    = App.create_button(frame, text="Reset graph", command=self.initialize, column=0, row=5, pady=(15,5), image=self.img_reset)
        self.reset_button   = App.create_button(frame, text="Reset",       command=self.initialize_plot, column=0, row=5, width=90, padx = (10,120), pady=(15,5))   # Multiplot button
        self.addplot_button = App.create_button(frame, text="Multiplot",   command=self.plot,            column=0, row=5, width=90, padx = (120,10), pady=(15,5))   # Multiplot button
        self.load_button    = App.create_button(frame, text="Load Folder", command=self.read_file_list,  column=0, row=1, image=self.img_folder)
        self.save_button    = App.create_button(frame, text="Save Figure/data", command=self.save_figure,     column=0, row=3,  image=self.img_save)
        self.set_button     = App.create_button(frame, text="Plot settings",command=lambda: self.open_toplevel(SettingsWindow, "Plot Settings"), column=0, row=4, image=self.img_settings)
        self.prev_button    = App.create_button(frame,                      command=lambda: self.change_plot(self.replot,-1), column=0, row=6, width=90, padx = (10,120), image=self.img_previous)
        self.next_button    = App.create_button(frame,                      command=lambda: self.change_plot(self.replot,1), column=0, row=6, width=90, padx = (120,10), image=self.img_next)
        
        #switches
        self.multiplot_button = App.create_switch(frame, text="Multiplot",  command=self.multiplot,   column=0, row=7, padx=20, pady=(10,5))
        self.uselabels_button = App.create_switch(frame, text="Use labels", command=self.use_labels,  column=0, row=8, padx=20)
        self.uselims_button   = App.create_switch(frame, text="Use limits", command=self.use_limits,  column=0, row=9, padx=20)
        self.fit_button       = App.create_switch(frame, text="Use fit",    command=lambda: self.open_fit_window(FitWindow), column=0, row=10, padx=20)
        self.normalize_button = App.create_switch(frame, text="Normalize",  command=self.normalize_setup,    column=0, row=11, padx=20)
        self.lineout_button   = App.create_switch(frame, text="Lineout",  command=self.lineout,    column=0, row=12, padx=20)
        self.FFT_button       = App.create_switch(frame, text="FFT",  command=self.fourier_trafo,    column=0, row=13, padx=20)
        self.subfolder_button = App.create_switch(frame, text="include subfolders", command=self.load_file_list, row=16, column=0, padx=20)
        
        #Data Table section
        self.data_table_button= App.create_switch(self.tabview.tab("Data Table"), text="Plot data table & custom data", command=None, row=0, column=0, padx=20)
        self.mid_axis_button  = App.create_switch(self.tabview.tab("Data Table"), text="middle axis lines", command=None, row=1, column=0, padx=20)
        self.yerror_area_button  = App.create_switch(self.tabview.tab("Data Table"), text="display y-error as shaded area", command=None, row=2, column=0, padx=20)
        self.data_table       = App.create_table(self.tabview.tab("Data Table"), width=300, sticky='ns', row=3, column=0, pady=(20,10), padx=10, rowspan=11)
        self.xval_min_entry   = App.create_entry(self.tabview.tab("Data Table"), row=0, column=3, width=90, text="x-limits", placeholder_text="x min")
        self.xval_max_entry   = App.create_entry(self.tabview.tab("Data Table"), row=0, column=4, width=90, placeholder_text="x max")
        self.datapoint_number_entry = App.create_entry(self.tabview.tab("Data Table"), row=1, column=3, width=90, text="length", placeholder_text="Default: 500")
        self.function_entry   = App.create_entry(self.tabview.tab("Data Table"), row=2, column=3, width=200, text="y-Function", columnspan=2, placeholder_text="np.sin(x)")

        self.column_dict = {}
        self.column_values = ["None", "x-values", "y-values", "x-error", "y-error"]
        for i in range(0,10):
            self.column_dict[f'column{i}'], self.column_dict[f'column{i}_label'] = App.create_Menu(self.tabview.tab("Data Table"), column=3, row= 3+i, width=100, values=self.column_values, columnspan=2, sticky='w',textwidget=True, text=f'column {i+1}')
        
        self.column_dict["column0"].set("x-values")
        self.column_dict["column1"].set("y-values")

        self.load_settings_frame()

        #tooltips
        self.tooltips = {"load_button": "Load a folder with data files\n- only files in this directory will be displayed in the list (no subfolders)\n- Standard path: Location of the Script or the .exe file\n- keybind: CTRL + O",
                    "save_button": "Save the current image or current data\n- Image file types: .png, .jpg, .pdf\n- Data file types: .dat, .txt, .csv\n- Save images in plain format (without the axis): 'Plot settings' -> 'Save plain images'\n- image format: 'Plot settings' -> 'Canvas Size'. (maximize app window)\n- keybind: CTRL + S",
                    "set_button":       "Open the Plot settings window\n- Top: Image Plot settings\n- Bottom Line Plot settings\n- Side: Boolean settings",
                    "plot_button":      "Reset and reinitialize the plot \n- keybind: CTRL + R",
                    "reset_button":     "Reset and reinitialize the plot \n- delete all previous multiplots\n- keybind: CTRL + R",
                    "addplot_button":   "Plot the currently selected file",
                    "prev_button":      "Plot the previous file in the list\n- keybind: CTRL + ←",
                    "next_button":      "Plot the next file in the list\n- keybind: CTRL + →",
                    "multiplot_button": "Create multiple plots in one figure \n- choose rows = 1, cols = 1 for single graph multiplots\n- Press CTRL+Z to delete the previous plot\n- You can click on the subplots to replot them or set the limits\n- keybind: CTRL + M",
                    "uselabels_button": "Create x,y-labels, add a legend and title\n- Legend settings: choose location, add file name to legend",
                    "uselims_button":   "Choose x,y-limits for line and image plots\n- x-limits stay the same when resetting the plot\n- y-limits will change to display all figures\n- the limits can be set with the sliders or typed in manually", 
                    "fit_button":       "Fit a function to the current data set\n- initial fit parameters can be set when the fit does not converge\n- use slider to fit only parts of the function\n- the textbox can be hidden via the 'Plot settings' -> 'Hide fit params'",
                    "normalize_button": "Normalize the plot\n- data file: normalize to max = 1 or area = 1 (Plot settings -> normalize Area)\n- image file: enhance the contrast of the image (can be changed manually: Plot settings -> pixel range)\n- keybind: CTRL + N", 
                    "lineout_button": "Display a lineout function\n- only used for images, no multiplot possible", 
                    "FFT_button": "Display the |FT(f)|² of the data\n- limits of the Fourier window can be set\n- zero padding (factor determines number of zeros/x-array length)\n- can only be used for line plots, Images don't work.\n- if the spectrum is given in nm, the FT will be calculated in fs",
                    "data_table_button": "Check this to plot own custom data in the window below\n- columns must be separated by a space or tab (not comma or semicolon)\n- decimal separator: point and comma works\n- You can also plot data by using numpy functions, the argument must be called 'x'.",
                    "ent_legend": "Add a Legend to the plot\n- Write '{name}' in the textbox if you want to add the file name to the legend.\n- The displayed file name can be customized in the legend settings.\n- You can access more data information with the following keys:\n+ {xmax},{ymax} - x,y coordinate of the maximum point\n+ {fwhm} - FWHM around the maximum point\n+ {x_res} - x-axis resolution\n- For files with multiple y-columns, the legend labels can be set by writing every label into a separate line\n- The legend for the second plot of a lineout or a FFT can be accessed with the second line of the legend entry",
                    "ent_subplot": "Specify the number of sub plots in each multiplot window\n- Not used for plotting in a single window (rows=1, cols=1)", 
                    "two_axis_button": "Initiate a second y-axis on the right\n- The x-axis is assumed to be the same\n- The button is reset to 'off' when a new multiplot window is created",
                    "legend_settings_button": "- Set the location of the legend\n- Customize the display of {name} by using a sub-string of the file name", 
                    "update_labels_button": "- Update the labels and titles\n - To update the legend, select the line by clicking on it (hide the line)",
                    "choose_ax_button": "Choose the axis, along which the lineout point will move",
                    "save_lineout_button": "Save the lineout into a data file",
                    "yerror_area_button": "off: Use errorbars with caps\non: Display the y-error as a shaded area above and below the line",
                    "mid_axis_button": "off: Use standard outside axis\non: Use middle line axis centered at (0,0)",
                    "function_entry": "Enter a mathematical function to display, Use numpy syntax\n List of possible functions:\n- np.sin(x), np.cos(x), np.tan(x)\n- np.arcsin(x), np.arccos(x)\n- np.deg2rad(x), np.rad2deg(x)\n- np.sinh(x), np.cosh(x), np.tanh(x)\n- np.exp(x), np.log(x), np.log10(x), np.log2(x)\n- np.i0(x) - Bessel function, np.sinc(x)\n- np.sqrt(x), np.cbrt(x), np.sign(x)\n- np.heaviside(x,0)",
                    "subfolder_button": "Display also files from any subfolder within the current directory"
                    }
        
        for name, description in self.tooltips.items():
            setattr(self, name+"_ttp", CreateToolTip(getattr(self, name), description))

        for name in ["multiplot_button", "uselabels_button", "uselims_button", "fit_button", "normalize_button", "lineout_button", "FFT_button"]:
                getattr(self, name).configure(state="disabled")

        
        # self.appearance_mode_label = App.create_label(frame, text="Appearance Mode:", row=17, column=0, padx=20, pady=(10, 0), sticky="w")
        self.appearance_mode_optionemenu = App.create_Menu(frame, values=["Appearance: Dark","Appearance: Light", "Appearance: System"], command=self.change_appearance_mode_event, width=200, row=18, column=0, padx=20, pady=(5,10))
        
        #entries
        self.folder_path = self.create_entry(column=2, row=0, text="Folder path", columnspan=2, width=600, padx=10, pady=10, sticky="w")
        self.folder_path.bind("<KeyRelease>", self.load_file_list)
        #dropdown menu
        self.optmenu = self.create_combobox(values=filelist, text="File name",row=1, column=2, columnspan=2, width=600, sticky="w", command=lambda x: self.determine_data_columns(x))
        if not filelist == []: self.optmenu.set(filelist[0])

    # initialize all widgets on the settings frame
    def load_settings_frame(self):
        self.settings_frame = customtkinter.CTkFrame(self, width=1, height=600, corner_radius=0)
        self.settings_frame.grid(row=0, column=4, rowspan=999, sticky="nesw")
        self.settings_frame.grid_columnconfigure(0, minsize=60)
        self.columnconfigure(2,weight=1)

        self.load_labels()
        self.load_multiplot()
        self.load_limits()
        self.load_lineout()
        self.load_fourier_trafo()

    def create_label(self, row, column, width=20, text=None, anchor='e', sticky='e', textvariable=None, padx=(5,5), image=None, pady=None, font=None, columnspan=1, fg_color=None, **kwargs):
        label = customtkinter.CTkLabel(self, text=text, textvariable=textvariable, width=width, image=image, anchor=anchor, font=font, fg_color=fg_color,  **kwargs)
        label.grid(row=row, column=column, sticky=sticky, columnspan=columnspan, padx=padx, pady=pady, **kwargs)
        return label

    def create_entry(self, row, column, width=200, height=28, text=None, columnspan=1, rowspan=1, padx=10, pady=5, placeholder_text=None, sticky='w', sticky_label='e', textwidget=False, init_val=None, **kwargs):
        entry = CustomEntry(self, width, height, placeholder_text=placeholder_text, **kwargs)
        entry.grid(row=row, column=column, columnspan=columnspan, rowspan=rowspan, padx=padx, pady=pady, sticky=sticky, **kwargs)
        if init_val is not None:
            entry.insert(0,str(init_val))
        if text is not None:
            entry_label = App.create_label(self, text=text, column=column-1, row=row, width=20, anchor='e', sticky=sticky_label)
            if textwidget == True:
                return (entry, entry_label)

        return entry

    def create_button(self, command, row, column, text=None, image=None, width=200, columnspan=1, padx=10, pady=5, sticky=None, **kwargs):
        button = customtkinter.CTkButton(self, text=text, command=command, width=width, image=image, **kwargs)
        button.grid(row=row, column=column, columnspan=columnspan, padx=padx, pady=pady, sticky=sticky, **kwargs)
        return button
    
    def create_segmented_button(self, values, command, row, column, width=200, columnspan=1, padx=10, pady=5, sticky=None, **kwargs):
        button = customtkinter.CTkSegmentedButton(self, values=values, command=command, width=width, **kwargs)
        button.grid(row=row, column=column, columnspan=columnspan, padx=padx, pady=pady, sticky=sticky, **kwargs)
        return button

    def create_switch(self, command, row, column, text, columnspan=1, padx=10, pady=5, sticky='w', **kwargs):
        switch = customtkinter.CTkSwitch(self, text=text, command=command, **kwargs)
        switch.grid(row=row, column=column, columnspan=columnspan, padx=padx, pady=pady, sticky=sticky, **kwargs)
        return switch
    
    def create_combobox(self, values, column, row, width=200, state='readonly', command=None, text=None, columnspan=1, padx=10, pady=5, sticky=None, **kwargs):
        combobox = customtkinter.CTkComboBox(self, values=values, command=command, state=state, width=width, **kwargs)
        combobox.grid(column=column, row=row, columnspan=columnspan, padx=padx, pady=pady, sticky=sticky, **kwargs)
        if text is not None:
            App.create_label(self, text=text, column=column-1, row=row, width=80, anchor='e',pady=pady)
        return combobox
    
    def create_Menu(self, values, column, row, command=None, text=None, width=200, columnspan=1, padx=10, pady=5, sticky=None, textwidget=False, init_val=None, **kwargs):
        optionmenu = customtkinter.CTkOptionMenu(self, values=values, width=width, command=command, **kwargs)
        optionmenu.grid(column=column, row=row, columnspan=columnspan, padx=padx, pady=pady, sticky=sticky, **kwargs)
        if init_val is not None:
            optionmenu.set(init_val)
        if text is not None:
            optionmenu_label = App.create_label(self, text=text, column=column-1, row=row, anchor='e', pady=pady)
            if textwidget == True:
                return (optionmenu, optionmenu_label)

        return optionmenu
    
    def create_slider(self, from_, to, row, column, width=200, text=None, init_val=None, command=None, columnspan=1, number_of_steps=1000, padx = 10, pady=5, sticky='w', textwidget=False,**kwargs):
        slider = customtkinter.CTkSlider(self, from_=from_, to=to, width=width, command=command, number_of_steps=number_of_steps)
        slider.grid(column=column, row=row, columnspan=columnspan, padx=padx, pady=pady, sticky=sticky, **kwargs)
        if text is not None:
            slider_label = App.create_label(self, text=text, column=column-1, row=row, width=20, anchor='e')
            if textwidget == True:
                return (slider, slider_label)
        if init_val is not None:
            slider.set(init_val)

        return slider
    
    def create_range_slider(self, from_, to, row, column, width=200, text=None, init_value=None, command=None, columnspan=1, number_of_steps=None, padx = 10, pady=5, sticky='w', textwidget=False,**kwargs):
        slider = CTkRangeSlider(self, from_=from_, to=to, width=width, command=command, number_of_steps=number_of_steps)
        slider.grid(column=column, row=row, columnspan=columnspan, padx=padx, pady=pady, sticky=sticky, **kwargs)
        if text is not None:
            slider_label = App.create_label(self, text=text, column=column-1, row=row, width=20, anchor='e')
            if textwidget == True:
                return (slider, slider_label)
        if init_value is not None:
            slider.set(init_value)

        return slider
    
    def create_table(self,  width, row, column, sticky=None, rowspan=1, **kwargs):
        text_widget = customtkinter.CTkTextbox(self, width = width, padx=10, pady=5)
        # text_widget.pack(fill="y", expand=True)
        text_widget.grid(row=row, column=column, sticky=sticky, rowspan=rowspan, **kwargs)
        self.grid_rowconfigure(row+rowspan-1, weight=1)

        return text_widget
    
    def create_textbox(self, row, column, width=200, height=28, text=None, columnspan=1, rowspan=1, padx=10, pady=5, sticky='w', sticky_label='e', textwidget=False, init_val=None, **kwargs):
        textbox = customtkinter.CTkTextbox(self, width, height, **kwargs)
        textbox.grid(row=row, column=column, columnspan=columnspan, rowspan=rowspan, padx=padx, pady=pady, sticky=sticky, **kwargs)
        if init_val is not None:
            textbox.insert(0,str(init_val))
        if text is not None:
            textbox_label = App.create_label(self, text=text, column=column-1, row=row, width=20, anchor='e', sticky=sticky_label)
            if textwidget == True:
                return (textbox, textbox_label)

        return textbox
    
    #############################################################
    ######################## Do the plots #######################
    #############################################################

    def initialize(self):
        if not self.initialize_plot_has_been_called:
            self.initialize_plot_has_been_called = True
            for name in ["multiplot_button", "uselabels_button", "uselims_button", "fit_button", "normalize_button", "lineout_button", "FFT_button"]:
                getattr(self, name).configure(state="enabled")

            self.fig = plt.figure(constrained_layout=True, dpi=150) 
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.tabview.tab("Show Plots"))
            self.canvas_widget = self.canvas.get_tk_widget()
            self.toolbar = self.create_toolbar()
            self.canvas_widget.pack(fill="both", expand=True)
            
            self.canvas.mpl_connect('pick_event', self.on_pick)
            self.canvas.mpl_connect("button_press_event", self.on_click)    # Choosing the axis by clicking on it


        if self.lineout_button.get():
            self.initialize_lineout(self.choose_ax_button.get())
        elif self.FFT_button.get():
            self.initialize_FFT()
        else:
            self.initialize_plot()

    def initialize_plot(self):
        self.fig.clear()
        self.canvas.draw()
        self.ax2 = None
        self.ax1 = None
        self.first_ax2 = True 
        self.plot_counter = 1      # counts how many plot-lines there are (counts all lines of all subplots) 
        self.plot_index = 0        # index of the currently used (sub)plot
        self.sub_plot_counter = 1  # number of subplots
        self.mouse_clicked_on_canvas = False
            
        if self.replot and not self.lineout_button.get() and not self.FFT_button.get():
            self.rows=float(self.ent_rows.get())
            self.cols=float(self.ent_cols.get())
        
        self.ax_container = []         # list of tuples containing (ax1,ax2) for every subplot
        self.multiplot_container = []  # list containing the "self.plot_container" for every previous subplot
        self.legend_container = [None] * int(self.rows * self.cols)
        self.plot_container = None     # plot container for current subplot: contains all lines ((x,y), fill, FWHM)
        self.legend = None

        self.ymax = float('-inf')
        self.ymin = float('inf')
        self.xmax = float('-inf')
        self.xmin = float('inf')

        self.plot()
        if ((self.uselims_button.get() and self.reset_plots and not self.lineout_button.get() and not self.FFT_button.get()) or (self.uselims_button.get() and self.image_plot != self.image_plot_prev)): 
            # make sure the sliders are reset
            self.update_slider_limits()
            xlim_min, xlim_max, ylim_min, ylim_max = self.reset_limits()

            if self.image_plot != self.image_plot_prev:
                self.xlim_slider.set([xlim_min, xlim_max])
                self.ylim_slider.set([ylim_min, ylim_max])
            
            self.update_plot("Update Limits")

    def plot(self):
        self.clicked_axis_index = None
        if self.replot and not self.lineout_button.get() and not self.FFT_button.get():
            self.sub_plot_value = float(self.ent_subplot.get())

        file_path = os.path.join(self.folder_path.get(), self.optmenu.get())

        # Decide if there is an image to process or a data file
        self.image_plot_prev = self.image_plot
        if file_path.endswith(image_type_names) and not self.data_table_button.get():
            self.image_plot = True 
        else:
            self.image_plot = False

        if self.image_plot: 
            if self.FFT_button.get(): self.FFT_button.toggle()
            self.lineout_button.configure(state="enabled")
            self.FFT_button.configure(state="disabled")
            if self.lineout_button.get():
                if self.fit_button.get(): self.fit_button.toggle()
                self.fit_button.configure(state="enabled")
            else: self.fit_button.configure(state="disabled")
        else:
            if self.lineout_button.get(): self.lineout_button.toggle()
            self.lineout_button.configure(state="disabled")
            self.FFT_button.configure(state="enabled")
            self.fit_button.configure(state="enabled")

        # multiplots but in several subplots
        if self.replot and (self.rows != 1 or self.cols != 1):
            self.ent_subplot.configure(state='normal')
            if self.sub_plot_counter >= self.sub_plot_value or self.plot_counter == 1:
                if self.plot_counter > self.rows*self.cols: return
                ax1 = self.fig.add_subplot(int(self.rows),int(self.cols),self.plot_counter)
                if self.plot_container != None: self.multiplot_container.append(self.plot_container)
                self.plot_container = [] 
                self.plot_order = []        # keeps track of the order the plots are created (one may switch between ax1 and ax2), needed for referencing
                self.legend = None
                if self.mouse_clicked_on_canvas:
                    self.ax1.remove()
                    self.ax_container[self.plot_counter-1] = (ax1, None)
                    self.mouse_clicked_on_canvas = False
                else: 
                    self.ax_container.append((ax1, None)) # Add primary axis with no twin yet
                self.ax1 = ax1
                self.ax2 = None
                self.sub_plot_counter = 1
                self.first_ax2 = True
                self.two_axis_button.deselect()
                self.plot_index = self.plot_counter-1
            else: 
                self.plot_counter -= 1
                self.sub_plot_counter += 1

        # single plot or multiplot but same axis
        elif ((self.replot and (self.rows == 1 and self.cols == 1)) or not self.replot) and self.ax_container == []:
            ax1 = self.fig.add_subplot(1,1,1)
            self.ax_container.append((ax1, None)) # Add primary axis with no twin yet
            self.ax1 = ax1
            self.plot_container = []
            self.plot_order = []
            self.ent_subplot.configure(state='disabled')

        # create the plot depending if we have a line or image plot
        if self.image_plot: 
            self.make_image_plot(file_path)
        else:
            colors = self.get_colormap(self.cmap)
            if (self.rows ==1 and self.cols==1):
                n = (self.plot_counter-1) % len(colors)  
            else:
                n = self.sub_plot_counter - 1
            self.ax1.set_prop_cycle(cycler(color=colors[n:]+ colors[:n]))
            self.data = self.load_plot_data(file_path)
            line_container = self.make_line_plot("data", "ax1")
            
        if self.uselabels_button.get() and (self.ent_title.get() != ""):
            self.fig.suptitle(self.ent_title.get()) 
        
        self.update_plot(None)


        if self.uselims_button.get() and not self.image_plot:
            self.ylim_slider.set([self.ymin, self.ymax])

        if not (self.rows == 1 and self.cols == 1):
            self.plot_counter = len(self.ax_container) + 1

    def get_colormap(self, cmap):  # for line plots not image plots

        if cmap == 'CUSTOM':
            colormap = ListedColormap([self.single_color])
            return colormap.colors
        
        colormap = plt.get_cmap(cmap)

        # Convert the continuous colormap into an array to pass into prop_cycle for line plots
        if cmap in sequential_colormaps:
            if cmap in ['Purples', 'Blues', 'Greens', 'Oranges', 'Reds']:
                # invert colormap and start at a brighter color (otherwise it almost looks black)
                colormap = ListedColormap(colormap(np.linspace(0.9, 0, self.cmap_length)))
            else:
                colormap = ListedColormap(colormap(np.linspace(0, 1, self.cmap_length)))
            colormap.colors = tuple(map(tuple, colormap.colors))

        return colormap.colors
        
    # set the axis labels, ticks, grid, middle axis line
    def plot_axis_parameters(self, ax, plot_counter=1):
        axis = getattr(self, ax)

        last_row = plot_counter >= self.cols*(self.rows-1)
        first_col = plot_counter % self.cols == 0
        # put the second axis on the right side of the plot
        if ax == "ax2":
            first_col = (plot_counter+self.cols) % self.cols == 0

        # if the hide_ticks option has been chosen in the multiplot, settings window
        if self.ticks_settings == "no ticks":
            axis.set_xticks([])
            axis.set_yticks([])
        elif "smart" in self.ticks_settings and not self.mid_axis_button.get():
            if not last_row: 
                axis.set_xticklabels([])
            if not first_col:
                axis.set_yticklabels([])

            if self.ticks_settings == "smart + tight" and self.label_settings == "smart":
                self.fig.get_layout_engine().set(w_pad=0, h_pad=0, hspace=0, wspace=0)
            else:
                self.fig.set_constrained_layout(True)

        if self.use_minor_ticks.get():
            axis.minorticks_on()
        else:
            axis.minorticks_off()

        if self.use_grid_lines.get():
            axis.grid(which="minor", color='0.97', alpha=0)
            axis.grid(which="major", color='0.85', alpha=0)

            alpha = 0.2 if self.image_plot else 1

            axis.grid(visible=True, which=self.grid_ticks, axis=self.grid_axis, alpha=alpha)
            axis.minorticks_on()
        else:
            axis.grid(visible=False)

        # place labels only on the left and bottom plots (if there are multiplots)
        if self.uselabels_button.get() == 1:
            if last_row or self.label_settings == "default" or self.mid_axis_button.get(): 
                axis.set_xlabel(self.ent_xlabel.get())
            if first_col or self.label_settings == "default" or self.mid_axis_button.get(): 
                axis.set_ylabel(self.ent_ylabel.get())

        if self.fit_button.get():
            if self.display_fit_params_in_plot.get():
                self.fit_params_box.set_visible(True)
            else: self.fit_params_box.set_visible(False)

        if self.mid_axis_button.get():
            axis.spines[['top', 'right']].set_visible(False)
            axis.spines[['left', 'bottom']].set_position('zero')
            axis.xaxis.set_tick_params(direction='inout')
            axis.yaxis.set_tick_params(direction='inout')
            axis.plot((1), (0), ls="", marker=">", ms=5, color="k",transform=axis.get_yaxis_transform(), clip_on=False)
            axis.plot((0), (1), ls="", marker="^", ms=5, color="k",transform=axis.get_xaxis_transform(), clip_on=False)
            xticks = axis.get_xticks().tolist()
            yticks = axis.get_yticks().tolist()
            if 0 in xticks: xticks.remove(0)
            if 0 in yticks: yticks.remove(0)
            axis.set_xticks(xticks)
            axis.set_yticks(yticks)
    
    # Determine the content of the columns in the data file
    def get_column_parameters(self, data_array):
        # Initialize the variables for plotting
        x_data, y_data, x_error, y_error = None, None, None, None
        y_data_list = []
        # Reverse the dictionary to map selections to column indices
        column_map = {"x-values": None, "y-values": [], "x-error": None, "y-error": None}

        # Populate the column_map based on the OptionMenu selections
        for key, widget in self.column_dict.items():
            if type(widget) is customtkinter.CTkOptionMenu:
                selected_value = widget.get()
                if selected_value == "y-values":
                    column_map["y-values"].append(int(key.replace('column', '')))
                else:
                    column_map[selected_value] = int(key.replace('column', ''))

        if column_map["x-values"] is not None:
            x_data = data_array[:, column_map["x-values"]]
        if column_map["x-error"] is not None:
            x_error = abs(data_array[:, column_map["x-error"]])
        if column_map["y-error"] is not None:
            y_error = abs(data_array[:, column_map["y-error"]])
        
        # Extract each y_data and y_error pair
        for i, y_column_index in enumerate(column_map["y-values"]):
            y_data = data_array[:, y_column_index]
            y_data_list.append(y_data)

        return x_data, y_data_list, x_error, y_error 

    # Make the legend for line plots
    def make_line_legend(self, axis, remove=True, container=None):
        if container is None:
            container = self.plot_container

        if self.ent_legend.get("0.0","end-1c") == "":
            return
        
        if self.legend is not None and remove == True and not self.FFT_button.get():
            self.legend.remove()

        if self.ax2 is not None:    
                handles1, labels1 = self.ax1.get_legend_handles_labels()
                handles2, labels2 = self.ax2.get_legend_handles_labels()

                handles, labels = [], []
                legend_map = {"ax1": (handles1, labels1), "ax2": (handles2, labels2)}

                # sort the handles based on the order in the self.plot_container which is tracked by self.plot_order
                for name in self.plot_order:
                    handle, label = legend_map[name]
                    handles.append(handle.pop(0))
                    labels.append(label.pop(0))

                self.legend = self.ax2.legend(handles, labels, fontsize=self.legend_font_size, fancybox=True, **self.legend_type)
        else:
            handles, labels = axis.get_legend_handles_labels()
            self.legend = axis.legend(handles, labels, fontsize=self.legend_font_size, fancybox=True, **self.legend_type)
        
        self.map_legend_to_ax = {}  # Will map legend lines to original lines.
        for legend_line, ax_line in zip(self.legend.legend_handles, [x for x in container]):
            legend_line.set_picker(5)  # Enable picking on the legend line.
            self.map_legend_to_ax[legend_line] = ax_line
        
        return self.map_legend_to_ax

    def make_line_plot(self, dat, ax):
        if self.multiplot_button.get():
                if self.two_axis_button.get():
                    if self.first_ax2:
                        ax2 = self.ax1.twinx()
                        ax2._get_lines = self.ax1._get_lines  # color cycle of ax1
                        self.ax_container[-1] = (self.ax1, ax2) # Update the container with the twin axis 
                        self.ax2 = ax2
                        self.first_ax2 = False
                    ax = "ax2"

        # Display 1D data
        if self.data.ndim == 1:
            self.data = np.vstack((range(len(self.data)), self.data)).T

        # Display data in the data table
        self.data_table.delete("0.0", "end")  # delete all text

        for row in self.data:
            self.data_table.insert("end", " \t ".join(format_number(val) for i, val in enumerate(row)) + "\n")

            
        data = getattr(self,dat)
        axis = getattr(self,ax)
        
        # hide unused column entries
        self.hide_column_entries(data)

        # create dictionary of the plot key word arguments
        self.plot_kwargs = dict(
            linestyle = self.linestyle,
            marker = self.markers,
            linewidth = self.linewidth,
            alpha = self.alpha
            )

        if self.normalize_button.get(): self.normalize()

        x_data, y_data_list, xerr, yerr = self.get_column_parameters(data)

        self.ymin = min(np.min(y_data_list), self.ymin)
        self.xmax = max(np.max(x_data), self.xmax)
        self.xmin = min(np.min(x_data), self.xmin)

        if not self.uselims_button.get():
            self.ymax = max(np.max(y_data_list), self.ymax)
        else:
            xmin_index = np.argmin(abs(x_data - self.xlim_slider.get()[0]))
            xmax_index = np.argmin(abs(x_data - self.xlim_slider.get()[1]))
            y_data_list_cropped = [y_data[xmin_index:xmax_index] for y_data in y_data_list]
            self.ymax = max(np.max(y_data_list_cropped), self.ymax)

        ######### create the plot
        plot = getattr(axis, "errorbar")
        text_lines = self.get_textbox_lines(self.ent_legend)
  
        for index, y_data in enumerate(y_data_list):
            if self.uselabels_button.get() and index < len(text_lines): 
                #self.ent_legend.get("0.0","end-1c")
                if ax == "ax_second" and len(text_lines) > 1: index = 1
                self.plot_kwargs["label"] = text_lines[index].format(name=self.optmenu.get()[self.legend_name], 
                                                                         ymax=np.max(y_data), 
                                                                         xmax=x_data[np.argmax(y_data)], 
                                                                         fwhm=find_fwhm(x_data,y_data)[-1], 
                                                                         res_x=x_data[1]-x_data[0])
            else: self.plot_kwargs["label"] = None

            ##### do the plot
            container = plot(x_data, moving_average(y_data, self.moving_average), xerr=xerr, yerr=yerr, capsize=3, **self.plot_kwargs)
            #####

            FWHM_line = ()
            fill = ()

            if self.draw_FWHM_line.get():
                x_min, x_max, y_value, _= find_fwhm(x_data,y_data)
                FWHM_line, = axis.plot([x_min,x_max], [y_value, y_value], color="black")
                
            if yerr is not None and self.yerror_area_button.get(): 
                fill = axis.fill_between(data[:,0], moving_average(y_data-yerr, self.moving_average), moving_average(y_data+yerr, self.moving_average), alpha=0.1)
                for item in list(flatten(container[1])): item.remove()
                for item in list(flatten(container[2])): item.remove()
                
            self.plot_container.append((container,fill, FWHM_line))
            self.plot_order.append(ax)
            self.plot_counter += 1


        axis.tick_params(which='both', direction='in')
        axis.set_yscale('log' if self.plot_type in ["semilogy", "loglog"] else 'linear')
        axis.set_xscale('log' if self.plot_type in ["semilogx", "loglog"] else 'linear')

        if self.uselabels_button.get(): 
            logging.info(f"Only important for multiplot. The current index of the subplot is: {self.plot_index}")
            self.legend_container[self.plot_index] = self.make_line_legend(axis)


        # create the fit
        if self.use_fit == 1 and not(self.FFT_button.get() and ax=="ax1") and (not self.image_plot or self.fit_button.get()):
            self.make_fit_plot(ax, dat)

        if self.FFT_button.get() and ax == "ax1":
            self.create_FFT_border_lines(ax)

        self.plot_axis_parameters(ax, plot_counter = self.plot_index)

        return container[0]
            
    def make_fit_plot(self, axis, data):
        self.create_fit_border_lines(axis)
        data = getattr(self,data)
        axis = getattr(self,axis)

          
        if self.FFT_button.get():
            center = data[np.argmax(abs(data[:,1])), 0]
            offset = abs(data[np.argmin(abs(data[:,1]-0.01*np.max(data[:,1]))),0]-center)
            xmin = center-offset
            xmax = center+offset
            xslider_min = xmin
            xslider_max = xmax
        else:
            xmin = np.min(data[:,0])
            xmax = np.max(data[:,0])
            if self.image_plot == self.image_plot_prev:
                xslider_min = max(xmin,self.fit_window.fit_borders_slider.get()[0])
                xslider_max = min(xmax,self.fit_window.fit_borders_slider.get()[1])
            else:
                xslider_min = xmin 
                xslider_max = xmax

        self.fit_window.fit_borders_slider.configure(from_=xmin, to=xmax)
        self.fit_window.fit_borders_slider.set([xslider_min, xslider_max])

        minimum = np.argmin(abs(data[:,0] - self.fit_window.fit_borders_slider.get()[0]))
        maximum = np.argmin(abs(data[:,0] - self.fit_window.fit_borders_slider.get()[1]))
        FitWindow.set_fit_params(self.fit_window, data[minimum:maximum,:])
        
        self.plot_kwargs_fit = dict(
            linestyle = "solid",
            marker = None,
            linewidth = self.linewidth
            )

        self.fit_lineout, = axis.plot(data[minimum:maximum, 0], self.fit_plot(self.function, self.params, data[minimum:maximum,:]), **self.plot_kwargs_fit)

        self.fit_params_box = axis.text(0.05, 0.9, FitWindow.get_fitted_labels(self.fit_window), color="black", ha='left', va='top', transform=axis.transAxes, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.6', alpha=0.8), size=8)

    def make_image_plot(self, file_path):
        # used in a onegraph multiplot when we change to an image
        # if self.image_plot != self.image_plot_prev and (self.rows == 1 and self.cols == 1) and self.replot:
        #     return

        # hide all column entries
        for entry in self.column_dict:
            self.column_dict[entry].grid_remove()

        # Read file as binary
        with open(file_path, 'rb') as f:
            image_data = f.read()

        # Convert to a NumPy array with np.frombuffer, Use io.BytesIO to simulate file opening and read it via matplotlib
        # self.data = plt.imread(io.BytesIO(np.frombuffer(image_data, np.uint8)))
        image = Image.open(io.BytesIO(image_data))

        # Check if the image is 16-bit (mode 'I;16') or 8-bit (mode 'L' or 'RGB')
        if image.mode in ('I;16', 'I'):
            # 16-bit image, normalize by 65535
            self.data = np.array(image).astype(np.float32) / 65535.0
        else:
            # 8-bit image, normalize by 255
            self.data = np.array(image).astype(np.float32) / 255.0

        # if the image has color channels, make sure they have the right order, if all channels are the same, reduce it to one channel
        if len(self.data.shape) == 3:

            r,g,b = self.data[:,:,0], self.data[:,:,1], self.data[:,:,2]
            if (b==g).all() and (b==r).all():
                self.data = self.data[...,0]
            elif self.convert_rgb_to_gray.get():
                self.data = 0.2989 * r + 0.5870 * g + 0.1140 * b
        
        self.data = np.flipud(self.data)

        # create dictionary of the plot key word arguments
        self.plot_kwargs = dict(
            cmap = self.cmap_imshow,
            interpolation = self.interpolation,
            aspect = self.aspect
            )

        if self.convert_pixels.get() == True:
            self.ax1.set(xticks=np.arange(0,len(self.data[0,:]), self.label_dist/self.pixel_size), xticklabels=['{:.1f}'.format(a) for a in np.arange(0,len(self.data[0,:])*self.pixel_size, self.label_dist)])
            self.ax1.set(yticks=np.arange(0,len(self.data[:,0]), self.label_dist/self.pixel_size), yticklabels=['{:.1f}'.format(a) for a in np.arange(0,len(self.data[:,0])*self.pixel_size, self.label_dist)])

        # normalize the image brightness
        if self.normalize_button.get(): 
            self.normalize()
        ######### create the plot
        self.image = self.ax1.imshow(self.data, origin='lower', clim=self.clim, **self.plot_kwargs)
        #########

        # use axis labels and add a legend
        if self.uselabels_button.get() and (self.ent_legend.get("0.0","end-1c") != ""): 
            text_lines = self.get_textbox_lines(self.ent_legend)
            if text_lines[0] != "": 
                # only take the first line as the label when doing lineout plots, otherwise take the whole entry
                label_text = text_lines[0] if self.lineout_button.get() else self.ent_legend.get("0.0","end-1c")   
                legend_label = label_text.format(name=self.optmenu.get()[self.legend_name])
                self.ax1.legend(labels=[legend_label], handles=[self.ax1.plot([],[])[0]], handlelength=0, handleheight=0, handletextpad=0, framealpha=1, fontsize=self.legend_font_size,fancybox=True,**self.legend_type)

        # use a scalebar on the bottom right corner
        if self.use_scalebar.get():
            scalebar = ScaleBar(self.pixel_size*1e-3, box_color="black", pad=0.4, box_alpha=.4, color="white", location="lower right", length_fraction=0.3, sep=2) # 1 pixel = 0.2 meter
            self.fig.gca().add_artist(scalebar)
        
        # use a colorbar
        if self.use_colorbar.get():
            self.cbar = self.fig.colorbar(self.image, pad=0.01)

        self.plot_axis_parameters("ax1", plot_counter = self.plot_index+1)
        
    def update_plot(self, val):
        ax = self.ax2 if self.two_axis_button.get() else self.ax1

        if self.uselims_button.get() == 1: 
            self.update_slider_limits()
            self.x_l, self.x_r = self.xlim_slider.get()
            self.y_l, self.y_r = self.ylim_slider.get()

            if isinstance(val, type(None)) and not self.image_plot:
                self.y_r = self.ymax

            if isinstance(val, (tuple, float, str, set, dict, type(None))):
                for (lim_box, limit) in zip(["xlim_lbox", "xlim_rbox", "ylim_lbox", "ylim_rbox"],
                                            ["x_l", "x_r", "y_l", "y_r"]):
                    box = getattr(self,lim_box)
                    if self.image_plot:
                        box.reinsert(0,str(int(getattr(self,limit))))
                    else:
                        box.reinsert(0,str(round(getattr(self,limit),4-len(str(int(getattr(self,limit)))))))

            if self.image_plot: # display images
                ax.set_xlim(left=float(self.x_l), right=float(self.x_r))
                ax.set_ylim(bottom=float(self.y_l), top=float(self.y_r))
                self.image.set_clim(self.clim)

            else: # line plot
                pad_val = 0.05 if not self.mid_axis_button.get() else 0
                ax.set_xlim(left=float(  self.x_l - pad_val * (self.x_r - self.x_l)), right=float(self.x_r + pad_val * (self.x_r - self.x_l)))
                ax.set_ylim(bottom=float(self.y_l - pad_val * (self.y_r - self.y_l)), top=float(  self.y_r + pad_val * (self.y_r - self.y_l)))
                
                if (self.plot_type == "semilogx" or self.plot_type == "loglog"):
                    xlim_l = calc_log_value(self.x_l - pad_val * (self.x_r - self.x_l), np.min(np.abs(self.data[:,0])), np.max(self.data[:,0]))
                    xlim_r = calc_log_value(self.x_r + pad_val * (self.x_r - self.x_l), np.min(np.abs(self.data[:,0])), np.max(self.data[:,0]))
                    ax.set_xlim(left=float(xlim_l), right=float(xlim_r))
                if (self.plot_type == "semilogy" or self.plot_type == "loglog"):
                    ylim_l = calc_log_value(self.y_l - pad_val * (self.y_r - self.y_l), np.min(np.abs(self.data[:,1])), np.max(self.data[:,1]))
                    ylim_r = calc_log_value(self.y_r + pad_val * (self.y_r - self.y_l), np.min(np.abs(self.data[:,1])), np.max(self.data[:,1]))
                    print(ylim_l, ylim_r, np.min(abs(self.data[:,1])), self.data[:,1])
                    ax.set_ylim(bottom=float(ylim_l), top=float(ylim_r))

        if self.fit_button.get():
            try:
                self.fit_xmin_line.set_xdata([self.fit_window.fit_borders_slider.get()[0]])
                self.fit_xmax_line.set_xdata([self.fit_window.fit_borders_slider.get()[1]])
            except:
                if self.lineout_button.get() or self.FFT_button.get():
                    self.create_fit_border_lines("ax_second")
                else:
                    self.create_fit_border_lines("ax1")
        
        if not self.image_plot:
            for i, line in enumerate(list(flatten(self.plot_container[-1][0]))):
                
                if line: 
                    line.set_alpha(self.alpha)
                    line.set_linewidth(self.linewidth)

                    if i == 0:  # line plot
                        line.set_linestyle(self.linestyle)
                    if i in [1,2]: # error caps
                        line.set_markeredgewidth(self.linewidth)

        self.canvas.draw()
    
    def load_plot_data(self, file_path):
        if self.data_table_button.get():
            data = self.read_table_data(self.data_table.get("0.0","end"))

            if self.function_entry.get() != "":
                xmin = float(self.xval_min_entry.get()) if self.xval_min_entry.get() else -5
                xmax = float(self.xval_max_entry.get()) if self.xval_max_entry.get() else 5
                length = int(self.datapoint_number_entry.get()) if self.datapoint_number_entry.get() else 500
                x = np.linspace(xmin, xmax,length)
                globals_dict = {"np": np, "x": x}
                y = eval(self.function_entry.get(), globals_dict)
                data = np.vstack((x,y)).T
        
        else:
            try:
                skip_rows = self.skip_rows(file_path)
                file_decimal = self.open_file(file_path)
                delimiter, maxcolumns = determine_delimiter_and_column_count(file_path, skip_rows)
                logging.info(f"skip rows = {skip_rows}, file decimal = {file_decimal}, delimiter = '{delimiter}', max columns = {maxcolumns}")

                # Custom converter to replace commas with dots, for numpy v>2.0.0, you need to remove ".decode('utf-8')"
                # comma_to_dot = lambda x: float(x.decode('utf-8').replace(file_decimal, '.'))
                comma_to_dot = lambda x: float(x.replace(file_decimal, '.'))

                # Load the data, applying the converter to all columns
                data = np.genfromtxt(file_path, skip_header=skip_rows, delimiter=delimiter,  filling_values=np.nan,  converters={i: comma_to_dot for i in range(maxcolumns)})
                # data = data[:, ~np.isnan(data).all(axis=0)]
            except Exception as error: 
                data = np.loadtxt(file_path)
                logging.error(f"np.genfromtxt was not succesful, fallback to np.loadtxt, Error = {error}")
        
        return data

    def create_fit_border_lines(self, axis):
        ax = getattr(self,axis)
        self.fit_xmin_line = ax.axvline([self.fit_window.fit_borders_slider.get()[0]], alpha = 0.4, lw=0.5)
        self.fit_xmax_line = ax.axvline([self.fit_window.fit_borders_slider.get()[1]], alpha = 0.4, lw=0.5)
    
    def create_FFT_border_lines(self, axis):
        ax = getattr(self,axis)
        self.FFT_xmin_line = ax.axvline([self.FFT_borders_slider.get()[0]], alpha = 0.4, lw=0.5)
        self.FFT_xmax_line = ax.axvline([self.FFT_borders_slider.get()[1]], alpha = 0.4, lw=0.5)

    def change_plot(self, replot, direction):
        index = filelist.index(self.optmenu.get()) + direction
        if 0 <= index < len(filelist):
            self.optmenu.set(filelist[index])
            self.reset_plots = False
            if replot and not self.lineout_button.get() and not self.FFT_button.get():
                self.plot()
            else:
                self.initialize()
            self.reset_plots = True

    def fit_plot(self, function, initial_params, data):
        params,K=cf(function,data[:, 0],data[:, 1], p0=initial_params)
        # write fit parameters in an array to access
        FitWindow.set_fitted_values(self.fit_window, params, K)
        return function(data[:,0], *params)
        
    def read_file_list(self):
        path = customtkinter.filedialog.askdirectory(initialdir=self.folder_path)
        if path != "":
            self.folder_path.reinsert(0, path)
        self.load_file_list()
        
    def load_file_list(self, val=None):
        global filelist
        if self.subfolder_button.get():
            filelist = natsorted(
            [ os.path.relpath(os.path.join(root, fname), start=self.folder_path.get())
                for root, _, files in os.walk(self.folder_path.get())
                for fname in files
                if fname.endswith(file_type_names)
            ])
        else:
            filelist = natsorted([fname for fname in os.listdir(self.folder_path.get()) if fname.endswith(file_type_names)])
        
        self.optmenu.configure(values=filelist)
        self.optmenu.set(filelist[0])
    
    def hide_column_entries(self, data):
        if data.ndim == 1: 
            return
        for i, entry in enumerate(self.column_dict):
            if i < 2*len(data[0,:]):
                self.column_dict[entry].grid()
            else:
                self.column_dict[entry].grid_remove()
                if type(self.column_dict[entry]) is customtkinter.CTkOptionMenu:
                    self.column_dict[entry].set(None)

    def determine_data_columns(self, file_name):
        if not file_name.endswith(image_type_names):
            data = self.load_plot_data(os.path.join(self.folder_path.get(), file_name))
            self.hide_column_entries(data)
        else:
            return

    def get_textbox_lines(self, textbox):
        line_text = []
        # Retrieve the number of lines in the Text widget
        num_lines = int(textbox.index('end-1c').split('.')[0])

        # Iterate over each line
        for i in range(1, num_lines + 1):
            line_text.append(textbox.get(f"{i}.0", f"{i}.end"))
        return line_text

    def control_z(self):
        
        if self.replot and (self.rows != 1 or self.cols != 1) and self.plot_counter > 2:
            self.plot_counter -= 1
            self.sub_plot_counter = float('inf')
            ax1, ax2 = self.ax_container.pop()
            ax1.remove()
            if ax2: ax2.remove()

            (self.ax1, self.ax2) = self.ax_container[-1]
        
        elif ((self.rows == 1 and self.cols == 1) and self.plot_counter > 1):
            # remove the line and possible caps and error-bar lines
            self.plot_container[-1][0].set_label("")
            for item in list(flatten(self.plot_container[-1])):
                try:
                    item.remove()
                except: pass

            self.plot_container.pop()
            self.plot_order.pop()

            # ensure that the color is setback to its previous state by shifting the colormap by the value n determined by the current plot counter and resetting the prop_cycle
            self.plot_counter -= 1

            if self.plot_counter == 1:  # needed to handle normalize correctly, when one plot is visible, and normalized is pressed, the plot has to be redrawn
                self.ax1.clear()
                self.ax1.set_xticks([])
                self.ax1.set_yticks([])
            if self.uselabels_button.get(): 
                self.legend_container[self.plot_index] = self.make_line_legend(self.ax1)

        self.canvas.draw()

    def read_table_data(self, data_string):
        data_string = data_string.replace(",",".")
        lines = data_string.strip().split('\n')
        # Initialize an empty list to hold the floats
        rows = []

        # Process each line
        for line in lines:
            # Split the line into individual numbers based on whitespace
            numbers = line.split()
            
            # Convert the numbers to floats and add to rows list
            rows.append([float(num) for num in numbers])

        # Convert the list of lists into a 2D NumPy array
        array_2d = np.array(rows)
        return array_2d

    ############################################################################################
    ##############################    Settings Frame     #######################################
    ############################################################################################

    def load_limits(self):
        row = 4 # 4 from the use_labels
        self.limits_title = App.create_label(self.settings_frame, text="Limits", font=customtkinter.CTkFont(size=16, weight="bold"), row=row, column=0, columnspan=5, padx=20, pady=(20, 5),sticky=None)
        self.labx = App.create_label(self.settings_frame, text="x limit", column=0, row=row+1)
        self.laby = App.create_label(self.settings_frame, text="y limit", column=0, row=row+2)

        for i, (lim_lbox, lim_rbox, lim_slider) in enumerate(zip(["xlim_lbox", "ylim_lbox"], # lim_box - entries
                                                                 ["xlim_rbox", "ylim_rbox"],
                                                                 ["xlim_slider", "ylim_slider"])): # lim_slider - slider 

            # create the entry objects "self.xlim_lbox" ...
            setattr(self, lim_lbox, App.create_entry(self.settings_frame, row=row+i+1, column=1, width=50))
            setattr(self, lim_rbox, App.create_entry(self.settings_frame, row=row+i+1, column=4, width=50))
            setattr(self, lim_slider, App.create_range_slider(self.settings_frame, from_=0, to=1, command= lambda val=None: self.update_plot(val), row=row+i+1, column =2, width=120, padx=(0,0), columnspan=2, init_value=[0,1]))

            # get the name of the created object, first the entry, second the slider
            entryl_widget = getattr(self,lim_lbox)
            entryr_widget = getattr(self,lim_rbox)
            slider_widget = getattr(self,lim_slider)

            #set the properties of the entry object, key release event, set the value to the slider and update the limits,
            # slider_widget = slider_widget is necessary to force the lambda function to capture the current value of slider_widget instead of calling it, when the key is pressed, otherwise slider_widget = ylim_r is used!
            entryl_widget.bind("<KeyRelease>", lambda event, val1=entryl_widget, val2=entryr_widget, slider_widget=slider_widget: (slider_widget.set([float(val1.get()), float(val2.get())]), self.update_plot(val1)))
            entryr_widget.bind("<KeyRelease>", lambda event, val1=entryl_widget, val2=entryr_widget, slider_widget=slider_widget: (slider_widget.set([float(val1.get()), float(val2.get())]), self.update_plot(val2)))
        
        self.use_limits()

    def use_limits(self):
        widget_names = ["xlim_slider","ylim_slider","labx","laby","limits_title", "xlim_lbox", "xlim_rbox", "ylim_lbox", "ylim_rbox"]

        if self.uselims_button.get() == 1:
            self.settings_frame.grid()
            [getattr(self, name).grid() for name in widget_names]
        else:
            [getattr(self, name).grid_remove() for name in widget_names]
            self.close_settings_window()

        if self.initialize_plot_has_been_called:
            self.update_slider_limits()
            xlim_min, xlim_max, ylim_min, ylim_max = self.reset_limits()
            self.xlim_slider.set([xlim_min, xlim_max])
            self.ylim_slider.set([ylim_min, ylim_max])
            self.update_plot("Update Limits")

    def reset_limits(self):
                # Slider
        if self.image_plot:
            xlim_min = 0
            ylim_min = 0
            xlim_max = len(self.data[0,:])
            ylim_max = len(self.data[:,0])
        
        else: # normal line plots
            xlim_min = np.min(self.data[:,0])
            ylim_min = np.min(self.data[:,1])
            xlim_max = np.max(self.data[:,0])
            ylim_max = np.max(self.data[:,1])
        
        return xlim_min, xlim_max, ylim_min, ylim_max
        
    #############################################################
    ######################## Labels #############################
    #############################################################

    def load_labels(self):
        row = 0
        self.labels_title = App.create_label(self.settings_frame, text="Labels", font=customtkinter.CTkFont(size=16, weight="bold"), row=row, column=0, columnspan=5, padx=20, pady=(20, 5),sticky=None)
        self.ent_ylabel   = App.create_entry(self.settings_frame,column=3, row=row+1, columnspan=2, width=110, placeholder_text="y label")
        self.ent_xlabel, self.ent_label_text   = App.create_entry(self.settings_frame,column=1, row=row+1, columnspan=2, width=110, placeholder_text="x label", text="x / y", textwidget=True)
        self.ent_legend, self.ent_legend_text  = App.create_textbox(self.settings_frame,column=1, row=row+2, columnspan=4, rowspan=2, width=110,text="Legend", textwidget=True)
        self.ent_title, self.ent_title_text  = App.create_entry(self.settings_frame,column=1, row=row+3, columnspan=4, width=110, placeholder_text="title",text="Title", textwidget=True)
        self.legend_settings_button = App.create_button(self.settings_frame, text="Settings", command=lambda: self.open_toplevel(LegendWindow, "Legend Settings"), column=3, row=row+2, columnspan=2, image=self.img_settings, width=110, sticky='w')
        self.update_labels_button = App.create_button(self.settings_frame, text="Update", command=self.update_labels, column=3, row=row+3, columnspan=2, image=self.img_reset, width=110, sticky='w')
        self.ent_legend.configure(border_width=2)
        self.use_labels()
        
    def use_labels(self):
        widget_names = ["ent_xlabel","ent_label_text","ent_ylabel","ent_legend","ent_legend_text","labels_title","legend_settings_button", "ent_title", "ent_title_text", "update_labels_button"]

        if self.uselabels_button.get():
            self.settings_frame.grid() 
            [getattr(self, name).grid() for name in widget_names]
            if self.show_title_entry.get():
                self.ent_legend.configure(height = 28)
                self.ent_legend.grid(rowspan=1)
            else:
                self.ent_legend.configure(height = 66)
                self.ent_legend.grid(rowspan=2)
                self.ent_title.grid_remove()
                self.ent_title_text.grid_remove()
        else:
            [getattr(self, name).grid_remove() for name in widget_names]
            self.close_settings_window()    
    
    def update_labels(self):
        ax = self.ax2 if self.two_axis_button.get() else self.ax1    
        ax.set_xlabel(self.ent_xlabel.get())
        ax.set_ylabel(self.ent_ylabel.get())
        self.fig.suptitle(self.ent_title.get())

        if self.image_plot and self.ent_legend.get("0.0","end-1c") != "":
            self.ax1.legend(labels=[self.ent_legend.get("0.0","end-1c")], handles=[self.ax1.plot([],[])[0]], handlelength=0, handleheight=0, handletextpad=0, framealpha=1, fontsize=self.legend_font_size,fancybox=True,**self.legend_type)
        elif self.clicked_axis_index is not None:
            plot_container = self.multiplot_container.copy()
            plot_container.append(self.plot_container)
            plot_container = plot_container[self.plot_index]

            if not plot_container[self.clicked_axis_index][0][0].get_visible():
                plot_container[self.clicked_axis_index][0].set_label(self.ent_legend.get("0.0","end-1c"))

            for ax_line in plot_container:
                ax_line = self.change_plot_visibility(ax_line, visibility=True)

                self.legend_container[self.plot_index] = self.make_line_legend(ax, remove=False, container=plot_container)
        self.canvas.draw()
    #############################################################
    ##################### multiplot #############################
    #############################################################
    
    def load_multiplot(self):
        row = self.multiplot_row
        self.multiplot_title = App.create_label( self.settings_frame,column=0, row=row, text="Multiplot Grid", font=customtkinter.CTkFont(size=16, weight="bold"), columnspan=5, padx=20, pady=(20, 5),sticky=None)
        self.ent_rows, self.ent_rows_text = App.create_entry( self.settings_frame,column=1, row=row+1, width=50, text="rows", textwidget=True)
        self.ent_cols, self.ent_cols_text = App.create_entry( self.settings_frame,column=3, row=row+1, width=50, text="columns", textwidget=True, columnspan=3, padx=(10,70))
        self.ent_subplot, self.ent_subplot_text = App.create_entry( self.settings_frame,column=1, row=row+2, width=50, text="subplots", textwidget=True)
        self.two_axis_button = App.create_switch(self.settings_frame, command=None, text="2nd axis", column=2, row=self.multiplot_row+2, columnspan=3, padx=20, pady=(10,5))
        self.multiplot()
        
    def multiplot(self):
        self.rows = 1
        self.cols = 1
        widget_names = ["ent_rows","ent_rows_text","ent_cols","ent_cols_text","reset_button","addplot_button","multiplot_title","ent_subplot","ent_subplot_text", "two_axis_button"]
        self.ent_subplot.reinsert(0,str(1))

        if self.multiplot_button.get() == 1:
            self.replot = True
            self.plot_button.grid_remove()
            self.settings_frame.grid()
            [getattr(self, name).grid() for name in widget_names]

            if self.image_plot:
                self.cols = 2
                self.rows = 2

            if self.lineout_button.get():
                self.cols = 2
                self.rows = 1

            self.ent_rows.reinsert(0,str(self.rows))
            self.ent_cols.reinsert(0,str(self.cols))
            
        else:
            self.replot = False
            self.plot_button.grid()
            [getattr(self, name).grid_remove() for name in widget_names]
            self.close_settings_window()

    #############################################################
    ######################## Lineout ############################
    #############################################################
            
    def load_lineout(self):
        row = 12
        self.lineout_title    = App.create_label( self.settings_frame, column=0, row=row, text="Lineout", font=customtkinter.CTkFont(size=16, weight="bold"), columnspan=5, padx=20, pady=(20, 5),sticky=None)
        self.choose_ax_button = App.create_Menu(self.settings_frame, column=4, row=row+2, values=[" x ", " y "], command=self.initialize_lineout, columnspan=2, width=60, padx=(0,10), sticky="w", init_val=" y ")
        self.save_lineout_button = App.create_button(self.settings_frame, column=4, row=row+1, image=self.img_save, command= lambda: self.save_data_file(self.lineout_data), width=60, padx=(0,10))
        self.angle_slider     = App.create_slider(self.settings_frame, from_=-90, to=90, command= lambda val=None: self.plot_lineout(val), row=row+1, column =2, width=120, padx=(0,0), columnspan=2, number_of_steps=180)
        self.line_slider      = App.create_slider(self.settings_frame, from_=0, to=1, command= lambda val=None: self.plot_lineout(val), row=row+2, column =2, width=120, padx=(0,0), columnspan=2)
        self.line_entry, self.ent_line_text = App.create_entry(self.settings_frame, row=row+2, column=1, width=50, text="line", textwidget=True)
        self.angle_entry, self.ent_angle_text = App.create_entry(self.settings_frame, row=row+1, column=1, width=50, text="angle", textwidget=True)
        self.angle_entry.insert(0,str(0))
        self.lineout()

    def lineout(self):
        if self.image_plot == False:
            self.lineout_button.deselect()

        widget_names = ["lineout_title", "line_entry", "line_slider", "ent_line_text", "choose_ax_button", "save_lineout_button", "angle_slider", "angle_entry", "ent_angle_text"]
        if self.lineout_button.get():
            self.display_fit_params_in_plot.set(False)
            if self.multiplot_button.get():
                self.multiplot_button.toggle()
            self.replot = True
            self.multiplot_button.configure(state="disabled")

            self.cols = 2
            self.rows = 1
            self.lineout_xvalue = int(len(self.data[0,:])/2)
            self.lineout_yvalue = int(len(self.data[:,0])/2)

            self.settings_frame.grid()
            [getattr(self, name).grid() for name in widget_names]
            
            self.line_entry.bind("<KeyRelease>", lambda event, val=self.line_entry, slider_widget=self.line_slider: (slider_widget.set(float(val.get())), self.plot_lineout(val)))
            self.line_slider.configure(to=len(self.data[0,:]))
            self.line_slider.set(self.lineout_yvalue)
            self.angle_entry.bind("<KeyRelease>", lambda event, val=self.angle_entry, slider_widget=self.angle_slider: (slider_widget.set(float(val.get())), self.plot_lineout(val)))
            self.initialize_lineout(self.choose_ax_button.get())
        else:
            [getattr(self, name).grid_remove() for name in widget_names]
            self.multiplot_button.configure(state="enabled")
            self.replot = False
            self.close_settings_window()

    def initialize_lineout(self,val):
        self.initialize_plot()
        if not self.lineout_button.get(): return

        if self.uselims_button.get():
            self.x_lineout_min = int(self.ylim_slider.get()[0])
            self.x_lineout_max = int(self.ylim_slider.get()[1])
            self.y_lineout_min = int(self.xlim_slider.get()[0])
            self.y_lineout_max = int(self.xlim_slider.get()[1])
        else:
            self.x_lineout_min = 0
            self.x_lineout_max = len(self.data[:,0])
            self.y_lineout_min = 0
            self.y_lineout_max = len(self.data[0,:])

        asp_image = len(self.data[self.x_lineout_min:self.x_lineout_max,0]) / len(self.data[0,self.y_lineout_min:self.y_lineout_max])

        if val == " x ":
            if self.line_slider.get() >= self.y_lineout_max: self.line_slider.set(self.y_lineout_max-1)
            self.lineout_data = np.array([self.data[int(self.line_slider.get()),self.x_lineout_min:self.x_lineout_max]]).T
            self.line_slider.configure(from_=self.y_lineout_min, to=self.y_lineout_max-1, number_of_steps = len(self.data[0,:])-1)
            
        elif val == " y ":
            if self.line_slider.get() >= self.x_lineout_max: self.line_slider.set(self.x_lineout_max-1)
            self.lineout_data = np.array([self.data[self.y_lineout_min:self.y_lineout_max,int(self.line_slider.get())]]).T
            self.line_slider.configure(from_=self.x_lineout_min, to=self.x_lineout_max-1, number_of_steps = len(self.data[:,0])-1)

        (x1, y1), (x2, y2) = self.get_lineout_data()
        self.axline = self.ax1.plot([x1, x2],[y1, y2], 'tab:blue')[0]
        self.axpoint = self.ax1.plot(self.lineout_xvalue, self.lineout_yvalue, 'tab:blue', marker="o")[0]
        
        # reset slider value when the lineout is changed and the value is out of bounds
        self.line_slider.set(self.line_slider.get())
        self.line_entry.reinsert(0,str(int(self.line_slider.get())))
        
        self.ax_second = self.fig.add_subplot(1,2,2)

        self.lineout_plot = self.make_line_plot("lineout_data", "ax_second")
        self.plot_axis_parameters("ax_second")
        self.ax_second.set_ylabel("")

        self.asp = np.diff(self.ax_second.get_xlim())[0]*asp_image
        logging.info(f"aspect = {self.asp}, asp image={asp_image},  {self.x_lineout_min}, {self.x_lineout_max}, {self.y_lineout_min}, {self.y_lineout_max}")

        self.update_lineout_aspect()

        self.canvas.draw()

    def get_lineout_data(self):
        if self.choose_ax_button.get() == " x ":
            self.lineout_xvalue = self.line_slider.get()
        elif self.choose_ax_button.get() == " y ":
            self.lineout_yvalue = self.line_slider.get()

        theta = -self.angle_slider.get() 
        (x1, y1), (x2, y2) = get_border_points(self.lineout_xvalue, self.lineout_yvalue, theta, (len(self.data[0,:]), len(self.data[:,0])))

        num = round(np.sqrt((x2-x1)**2 + (y2-y1)**2))
        x, y = np.linspace(x1, x2, num), np.linspace(y1, y2, num)

        # Extract the values along the line, using cubic interpolation
        brightness_value = scipy.ndimage.map_coordinates(self.data, np.vstack((y,x)))
        scale = self.pixel_size if self.convert_pixels.get() else 1

        self.lineout_data = np.vstack([np.arange(0,num*scale,scale), brightness_value]).T
        return (x1, y1), (x2, y2)

    def plot_lineout(self, val):
        self.line_entry.reinsert(0,str(int(self.line_slider.get())))
        self.angle_entry.reinsert(0,str(int(self.angle_slider.get())))
        
        current_axis = self.choose_ax_button.get()
        if self.angle_slider.get() in [-90, 90]: self.choose_ax_button.set(" x ")
        if self.angle_slider.get() == 0:         self.choose_ax_button.set(" y ")
        if current_axis != self.choose_ax_button.get():
            self.initialize_lineout(self.choose_ax_button.get())

        (x1, y1), (x2, y2) = self.get_lineout_data()

        self.axline.set_data([x1, x2],[y1, y2])
        self.axpoint.set_data([self.lineout_xvalue], [self.lineout_yvalue])
        self.lineout_plot.set_data(self.lineout_data[:,0], self.lineout_data[:,1])
        self.update_lineout_aspect()

        if self.use_fit == 1:
            minimum = np.argmin(abs(self.lineout_data[:,0] - self.fit_window.fit_borders_slider.get()[0]))
            maximum = np.argmin(abs(self.lineout_data[:,0] - self.fit_window.fit_borders_slider.get()[1]))
            self.fit_lineout.set_xdata(self.lineout_data[minimum:maximum,0])
            self.fit_lineout.set_ydata(self.fit_plot(self.function, self.params, self.lineout_data[minimum:maximum,:]))
        self.canvas.draw()

    def update_lineout_aspect(self):
        self.ax_second.set_aspect(abs(self.asp)/(self.clim[1]-self.clim[0]))
        self.ax_second.set_ylim(self.clim)
    
    #############################################################
    ############### Fouriertransformation #######################
    #############################################################
        
    def load_fourier_trafo(self):
        row = 15
        self.FFT_title    = App.create_label( self.settings_frame,column=0, row=row, text="Fourier Transform", font=customtkinter.CTkFont(size=16, weight="bold"), columnspan=5, padx=20, pady=(20, 5),sticky=None)
        self.save_FFT_button = App.create_button(self.settings_frame, column = 3, row=row+1, text="Save FFT", command= lambda: self.save_data_file(self.FFT_data), width=80, columnspan=2)
        self.padd_zeros_textval  = App.create_label( self.settings_frame,column=1, columnspan=2, row=row+1, text=str(0), sticky="n", anchor="n", fg_color="transparent")
        self.padd_zeros_slider, self.padd_zeros_label = App.create_slider(self.settings_frame, from_=0, to=10, command= self.update_FFT, row=row+1, column=1, columnspan=2, init_val=0, number_of_steps=10, width=120, text="padd 0's", padx=(10,0), textwidget=True)
        self.FFT_borders_slider, self.FFT_borders_label = App.create_range_slider(self.settings_frame, from_=0, to=1, command= self.update_FFT, row=row+2, column =1, width=120, columnspan=2, text="FFT lims", padx=(10,0), textwidget=True)
        self.fourier_trafo()
            
    def fourier_trafo(self):
        row = 15
        widget_names = ["FFT_title", "save_FFT_button", "FFT_borders_slider", "padd_zeros_textval", "padd_zeros_slider", "padd_zeros_label", "FFT_borders_label"]

        if self.image_plot == True:
            self.FFT_button.deselect()

        if self.FFT_button.get():
            self.display_fit_params_in_plot.set(False)
            if self.multiplot_button.get():
                self.multiplot_button.toggle()
            if self.lineout_button.get():
                self.lineout_button.toggle()
            self.replot = True
            self.multiplot_button.configure(state="disabled")
            self.lineout_button.configure(state="disabled")
            self.settings_frame.rowconfigure(row+1, minsize=50)

            self.cols = 2
            self.rows = 1
            self.settings_frame.grid()
            [getattr(self, name).grid() for name in widget_names]

            self.FFT_borders_slider.configure(from_= np.min(self.data[:,0]), to=np.max(self.data[:,0]))
            self.FFT_borders_slider.set([np.min(self.data[:,0]),np.max(self.data[:,0])])
            
            self.initialize_FFT()
        else:
            [getattr(self, name).grid_remove() for name in widget_names]
            self.multiplot_button.configure(state="enabled")
            self.lineout_button.configure(state="enabled")
            self.settings_frame.rowconfigure(row+1, minsize=0)
            self.replot = False
            self.close_settings_window()

    def initialize_FFT(self):
        self.initialize_plot()
        if self.FFT_button.get() == False: return
        
        # ensure that the interpolated spectrum looks the same
        # self.ax1.plot(3e8/(freq_points*1e-9), interp_spec)
        self.ax_second = self.fig.add_subplot(1,2,2)

        self.FFT_data = self.Fourier_transform("data")
        self.FFT_plot = self.make_line_plot("FFT_data", "ax_second")

        # Determine the width of the displayed FT window
        argmax = np.argmax(self.FFT_data[:,1])
        width = np.max(np.argwhere(self.FFT_data[:,1] > 1e-3*np.max(self.FFT_data[:,1]))) - argmax + int(0.01*len(self.FFT_data[:,1])/(self.padd_zeros_slider.get()+1))
        self.ax_second.set_xlim(self.FFT_data[argmax-width, 0], self.FFT_data[argmax+width, 0])

        self.canvas.draw()
    
    def Fourier_transform(self, dat):
        dat = getattr(self, dat)
        data = dat[np.argmin(abs(self.FFT_borders_slider.get()[0]-dat[:,0])):np.argmin(abs(self.FFT_borders_slider.get()[1]-dat[:,0])),:]
        freq = 3e8/(data[:,0]*1e-9)
        interp_func = interp1d(freq, data[:,1], kind='linear', bounds_error = False, fill_value=0)
        freq_points = np.linspace(freq.min(),freq.max(), len(data[:,0]))
        interp_spec = interp_func(freq_points)
        Delta_w = (freq_points[1]-freq_points[0])
        padding = int(self.padd_zeros_slider.get()*len(data[:,0])/2)
        interp_spec = np.pad(interp_spec, (padding,padding))
        freq_points = np.pad(freq_points, (padding,padding), mode='linear_ramp', end_values=(freq_points[0]-padding*Delta_w,freq_points[-1]+padding*Delta_w))

        FFT_data = abs(np.fft.fft(np.sqrt(abs(interp_spec))))**2
        FFT_data = np.fft.fftshift(FFT_data)
        FFT_freq = np.fft.fftfreq(freq_points.size, d = 1e-15*Delta_w)
        FFT_freq = np.fft.fftshift(FFT_freq)
        FFT_data = np.vstack((FFT_freq, FFT_data)).T
        return FFT_data

    def update_FFT(self, val):
        try:
            self.FFT_xmin_line.set_xdata([self.FFT_borders_slider.get()[0]])
            self.FFT_xmax_line.set_xdata([self.FFT_borders_slider.get()[1]])
        except:
            self.create_FFT_border_lines("ax1")

        self.padd_zeros_textval.configure(text=str(self.padd_zeros_slider.get()))
        self.FFT_data = self.Fourier_transform("data")
        self.FFT_plot.set_xdata(self.FFT_data[:,0])
        self.FFT_plot.set_ydata(self.FFT_data[:,1])
        if self.use_fit == 1:
            minimum = np.argmin(abs(self.FFT_data[:,0] - self.fit_window.fit_borders_slider.get()[0]))
            maximum = np.argmin(abs(self.FFT_data[:,0] - self.fit_window.fit_borders_slider.get()[1]))
            self.fit_lineout.set_xdata(self.FFT_data[minimum:maximum,0])
            self.fit_lineout.set_ydata(self.fit_plot(self.function, self.params, self.FFT_data[minimum:maximum,:]))
        self.canvas.draw()

    #############################################################
    ##################### Misc. Functions #######################
    #############################################################

    def save_data_file(self, data):
        file_name = customtkinter.filedialog.asksaveasfilename()
        if file_name != "":
            np.savetxt(file_name, data, fmt="%.4e")
        
    def update_slider_limits(self):
        if self.image_plot:
            self.xlim_slider.configure(from_=0, to=len(self.data[0,:]))
            self.ylim_slider.configure(from_=0, to=len(self.data[:,0]))
        else:
            self.xlim_slider.configure(from_=self.xmin-(self.xmax-self.xmin)*0.2, to=self.xmax+(self.xmax-self.xmin)*0.2)  
            self.ylim_slider.configure(from_=self.ymin-(self.ymax-self.ymin)*0.2, to=self.ymax+(self.ymax-self.ymin)*0.2)      

    def normalize_setup(self):
        if self.normalize_button.get():
            self.normalize()
        else: self.clim = (0,1)

        if self.image_plot: 
            self.image.set_clim(self.clim)
        else:
            self.control_z()
            self.plot()
        self.canvas.draw()

    def normalize(self):
        if self.image_plot:
            # self.data = self.data * 0.5/np.mean(self.data)*self.enhance_value
            # self.data[self.data>1] = 1
            pixel_values = self.data.flatten()
            
            # Calculate the lower and upper percentile for 98% of the pixel values
            lower_bound = np.percentile(pixel_values, 1)
            upper_bound = np.percentile(pixel_values, 99)

            self.clim = (lower_bound, upper_bound)
        elif self.normalize_function == "Maximum":
            self.data[:, 1:] /= np.max(self.data[:, 1]) / self.normalize_value
        elif self.normalize_function == "Area":
            self.data[:, 1:] /= (np.sum(self.data[:, 1]) * abs(self.data[0,0]-self.data[1,0])) / self.normalize_value
        elif self.normalize_function == "Only Factor":
            self.data[:, 1:] *= self.normalize_value

        self.ymax = max([self.ymax, np.max(self.data[:, 1])])

    def create_toolbar(self) -> customtkinter.CTkFrame:
        # toolbar_frame = customtkinter.CTkFrame(master=self.tabview.tab("Show Plots"))
        # toolbar_frame.grid(row=1, column=0, sticky="ew")
        toolbar_frame = customtkinter.CTkFrame(self)
        toolbar_frame.grid(row=20, column=1, columnspan=2, padx=(10), sticky="ew")
        toolbar = CustomToolbar(self.canvas, toolbar_frame)
        toolbar.config(background=self.color)
        toolbar._message_label.config(background=self.color, foreground=self.text_color, font=(15))
        toolbar.winfo_children()[-2].config(background=self.color)
        toolbar.update()
        return toolbar_frame
    
    def toggle_toolbar(self, value):
        if not self.initialize_plot_has_been_called: return
        if value == "Show Plots":
            self.toolbar.grid()
        else:
            self.toolbar.grid_remove()

    # Check if there are any non-number characters in the file and skip these lines
    def skip_rows(self, file_path):
        skiprows = 0
        for line in open(file_path, 'r'):
            if line[0].isdigit() or (line[0] == " " and line[1].isdigit()) or (line[0] == "-" and line[1].isdigit()):
                break
            else:
                skiprows += 1
        if skiprows > 0: logging.info(f"There were {skiprows} head lines detected and skipped.")
        return skiprows

    # Save the current figure or the data based on the file type
    def save_figure(self):
        file_name = customtkinter.filedialog.asksaveasfilename()
        if (self.image_plot and self.save_plain_image.get() and file_name.endswith((".pdf",".png",".jpg",".jpeg",".PNG",".JPG",".svg"))):
            if self.uselims_button.get(): 
                plt.imsave(file_name, np.flipud(self.data[int(self.y_l):int(self.y_r),int(self.x_l):int(self.x_r)]), cmap=self.cmap_imshow)
            else: 
                plt.imsave(file_name, np.flipud(self.data), cmap=self.cmap_imshow, vmin= self.clim[0], vmax=self.clim[1])
        elif file_name.endswith((".pdf",".png",".jpg",".jpeg",".PNG",".JPG",".svg")): 
            self.fig.savefig(file_name, bbox_inches='tight')
        elif file_name.endswith((".dat",".txt",".csv")):
            if self.image_plot:
                np.savetxt(file_name, self.data.astype(int), fmt='%i')
            else:
                np.savetxt(file_name, self.data)

    # Open a file and count the number of . and , to decide the delimiter
    def open_file(self, file_path) -> str:
        try:
            with open(file_path, 'r') as file:
                content = file.read()
        except Exception as e:
            logging.error(f"Error reading the file: {e}")
            return

        if content.count('.') >= content.count(','):
            return '.'
        elif content.count('.') < content.count(','):
            return ','
        else:
            logging.warning("No decimal separators (commas or dots) were found in the file.")
    
    def change_appearance_mode_event(self, new_appearance_mode: str):
        new_appearance_mode = new_appearance_mode.replace("Appearance: ", "")
        customtkinter.set_appearance_mode(new_appearance_mode)
        if new_appearance_mode == "Light":
            self.color = "#e5e5e5"
            self.text_color = "black"
        else:
            self.color = "#212121"
            self.text_color = "white"
        if self.initialize_plot_has_been_called: 
            self.toolbar.pack_forget()
            self.toolbar = self.create_toolbar()

    def toggle_boolean(self, boolean):
        boolean.set(not boolean.get())

    # Close settings window if it is no longer needed
    def close_settings_window(self):
        if not (self.uselabels_button.get() or self.uselims_button.get() or self.fit_button.get() or self.multiplot_button.get() or self.lineout_button.get() or self.FFT_button.get()):
            self.settings_frame.grid_remove()
            
    def open_toplevel(self,cls,var):
        if self.toplevel_window[var] is None or not self.toplevel_window[var].winfo_exists():
            self.toplevel_window[var] = cls(self)  # create window if its None or destroyed

        self.toplevel_window[var].focus()  # focus it
    
    # Open the fit window Menu
    def open_fit_window(self,cls):
        if self.fit_button.get() == 1:
            if self.image_plot and not self.lineout_button.get(): 
                self.fit_button.deselect()
            else:   
                self.fit_window = cls(self)  # create window if its None or destroyed
                self.initialize()
        else:
            self.fit_window.close_window()
            self.close_settings_window()
            self.fit_window.destroy()
            self.fit_xmin_line.remove()
            self.fit_xmax_line.remove()

    # Closing the application       
    def on_closing(self):
        for name, description in self.tooltips.items():
            tooltip = getattr(self, name+"_ttp")
            tooltip.cleanup()

        self.quit()    # Python 3.12 works
        self.destroy() # needed for built exe

    # Click on a subplot to select it as the current plot
    def on_click(self, event):
        if self.rows == 1 and self.cols == 1:
            return

        for i, (ax1,ax2) in enumerate(self.ax_container):
            if ax1.in_axes(event):
                # highlighting the subplot
                ax1.set_facecolor('0.9')
                self.ax1 = ax1 
                self.ax2 = ax2
                self.plot_counter = i+1
                self.plot_index = i
                self.mouse_clicked_on_canvas = True
                if self.uselabels_button.get() and self.update_labels_multiplot.get():
                    self.ent_xlabel.reinsert(0, self.ax1.xaxis.get_label().get_text())  # reinsert the old x- and y-labels
                    self.ent_ylabel.reinsert(0, self.ax1.yaxis.get_label().get_text())
                    # try:
                    #     _, label = self.ax1.get_legend_handles_labels()
                    #     self.ent_legend.reinsert(0, label[0])
                    # except:
                    #     pass
                self.plot_axis_parameters
            else:
                ax1.set_facecolor('white')
        
        logging.info(event)
        logging.info(f"currently clicked subplot: {self.plot_index} (Info located in on_click())")
        # if no axes are clicked
        if event.inaxes == None:
            self.plot_counter = len(self.ax_container) + 1
            self.mouse_clicked_on_canvas = False
        self.canvas.draw_idle()

    # Pick and hide a specific plot line
    def on_pick(self, event):
        # On the pick event, find the original line corresponding to the legend
        # proxy line, and toggle its visibility.
        legend_line = event.artist

        logging.info(f"subplot index of the currently used plot: {self.plot_index} (Info located in on_pick())")
        map_legend_to_ax = self.legend_container[self.plot_index]
        plot_container = self.multiplot_container.copy()
        plot_container.append(self.plot_container)

        # Do nothing if the source of the event is not a legend line.
        if legend_line not in map_legend_to_ax:
            return

        ax_line = map_legend_to_ax[legend_line]
        visible = not ax_line[0][0].get_visible()
        ax_line = self.change_plot_visibility(ax_line, visible)

        # get the index of the clicked axis, this is important to reset the label of this axis
        self.clicked_axis_index = next((i for i, plot in enumerate(plot_container[self.plot_index]) if plot is ax_line), None)
        logging.info(f"Index of the currently selected line in the subplot: {self.clicked_axis_index} (Info located in on_pick())")

        # Change the alpha on the line in the legend, so we can see what lines have been toggled.
        legend_line.set_alpha(1.0 if visible else 0.2)
        self.canvas.draw() 
    
    def change_plot_visibility(self, ax_line, visibility):
        for a in ax_line[0].get_children():
            a.set_visible(visibility)
        
        for a in ax_line[1:]:
            if a: a.set_visible(visibility)

        return ax_line
"""
############################################################################################
##############################    Settings Window    #######################################
############################################################################################

"""

class SettingsWindow(customtkinter.CTkToplevel): 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry("600x730")
        self.title("Plot Settings")
        self.app = app

        # general values
        self.canvas_ratio = {'Auto': None, 'Custom': 0,'4:3 ratio': 4/3, '16:9 ratio': 16/9, '3:2 ratio': 3/2, '3:1 ratio': 3,'2:1 ratio': 2, '1:1 ratio': 1, '1:2 ratio': 0.5}
        self.ticks_settings = ["smart", "smart + tight", "default", "no ticks"]
        self.label_settings = ["smart", "default"]
        # values for image plots
        self.cmap_imshow = ['magma','hot','viridis', 'plasma', 'inferno', 'cividis', 'gray', 'bone', 'afmhot', 'copper']
        self.aspect_values = ['equal', 'auto']
        
        # values for line plots
        self.linestyle = {'solid': '-', 'dashed': '--', 'dotted': ':', 'dashdotted': '-.', 'no line': ''}
        self.markers = {'none':'', 'point':'.', 'circle':'o', 'pixel':',', '∇': 'v', 'Δ': '^', 'x': 'x', '◆': 'D'}
        self.grid_lines = ['major','minor','both']
        self.grid_axis = ['both', 'x', 'y']
        self.cmap = ['tab10', 'tab20', 'tab20b','tab20c', 'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3', 'CUSTOM']
        self.cmap.extend(sequential_colormaps)
        self.moving_average = ['1','2','4','6','8','10','16']
        self.cmap_length = ['5','10','15','20','25','30','35','40']
        self.single_colors = {'blue':'tab:blue','orange':'tab:orange','green':'tab:green','red':'tab:red','purple':'tab:purple','brown':'tab:brown','pink':'tab:pink','gray':'tab:gray','olive':'tab:olive','cyan':'tab:cyan'}
        self.plot_type = {'Linear': 'errorbar', 'Semi Logarithmic x': 'semilogx', 'Semi Logarithmic y': 'semilogy', 'Log-Log plot': 'loglog'}
        self.normalize_function = ['Maximum', 'Area', 'Only Factor']

        App.create_label(self, column=1, row=0, columnspan=2, text="General settings", font=customtkinter.CTkFont(size=16, weight="bold"),padx=20, pady=(20, 5), sticky=None)
        App.create_label(self, column=1, row=4, columnspan=2, text="Image plot settings", font=customtkinter.CTkFont(size=16, weight="bold"),padx=20, pady=(20, 5), sticky=None)
        App.create_label(self, column=1, row=10, columnspan=2, text="Line plot settings", font=customtkinter.CTkFont(size=16, weight="bold"),padx=20, pady=(20, 5), sticky=None)

        # general values 
        self.canvas_ratio_list   = App.create_Menu(self, column=1, row=1, width=110, values=list(self.canvas_ratio.keys()), text="Canvas Size", command=lambda x: self.update_canvas_size(self.canvas_ratio[x]))
        self.canvas_width        = App.create_entry(self,column=2, row=1, width=70, placeholder_text="10 [cm]", sticky='w', init_val=self.app.canvas_width)
        self.canvas_height       = App.create_entry(self,column=3, row=1, width=70, placeholder_text="10 [cm]", sticky='w', init_val=self.app.canvas_height, columnspan=2)
        self.ticks_settings_list = App.create_Menu(self, column=1, row=2, width=110, values=self.ticks_settings, text="Ticks, Label settings", command=self.apply_settings, init_val=self.app.ticks_settings)
        self.label_settings_list = App.create_Menu(self, column=2, row=2, width=70, values=self.label_settings, command=self.apply_settings, init_val=self.app.label_settings)

        # values for image plots
        self.pixel_size         = App.create_entry(self,column=1, row=5, columnspan=2, text="Pixel size [µm]", placeholder_text="7.4 µm", sticky='w', init_val=self.app.pixel_size*1e3)
        self.label_dist         = App.create_entry(self,column=1, row=6, columnspan=2, text="Label step [mm]", placeholder_text="1 mm", sticky='w', init_val=self.app.label_dist)
        self.aspect             = App.create_Menu(self, column=1, row=7, columnspan=2, values=self.aspect_values, text="Aspect",  command=self.apply_settings, init_val=self.app.aspect)
        self.cmap_imshow_list   = App.create_Menu(self, column=1, row=8, columnspan=2, values=self.cmap_imshow, text="Colormap",command=self.apply_settings, init_val=self.app.cmap_imshow)
        self.clim_slider     = App.create_range_slider(  self,   column=1, row=9, columnspan=2, from_=0, to=1, width=155,
                                                    command= lambda value, var="pixel_range_var": self.update_rangeslider_value(value, var), text="Pixel range", init_value=list(self.app.clim))

        # values for line plots
        self.plot_type_list     = App.create_Menu(self, column=1, row=11, columnspan=2, values=list(self.plot_type.keys()),text="Plot type",command=self.apply_settings)
        self.linestyle_list     = App.create_Menu(self, column=1, row=12, width=110, values=list(self.linestyle.keys()), text="Line style & Marker", command=self.apply_settings)
        self.marker_list        = App.create_Menu(self, column=2, row=12, width=70, values=list(self.markers.keys()),     command=self.apply_settings)
        self.cmap_list          = App.create_Menu(self, column=1, row=13, width=110, values=self.cmap, text="Colormap", command=self.apply_settings, init_val=self.app.cmap)
        self.single_colors_list = App.create_Menu(self, column=2, row=13, width=70, values=list(self.single_colors.keys()), command=self.apply_settings)
        self.cmap_length_list   = App.create_Menu(self, column=2, row=13, width=70, values=self.cmap_length, command=self.apply_settings, init_val=self.app.cmap_length)
        self.grid_lines_list    = App.create_Menu(self, column=1, row=14, width=110, values=self.grid_lines, text="Grid",  command=self.apply_settings, init_val=self.app.grid_ticks)
        self.grid_axis_list     = App.create_Menu(self, column=2, row=14, width=70, values=self.grid_axis, command=self.apply_settings, init_val=self.app.grid_axis)
        self.normalize_list     = App.create_Menu(self, column=1, row=15, width=110, values=self.normalize_function, text="Normalize",  command=self.apply_settings, init_val=self.app.normalize_function)
        self.normalize_value    = App.create_entry(self,column=2, row=15, width=70, columnspan=2, sticky='w', init_val=self.app.normalize_value)
        self.moving_av_list     = App.create_Menu(self, column=1, row=16, columnspan=2, values=self.moving_average, text="Average", command=self.apply_settings, init_val=self.app.moving_average)
        self.linewidth_slider   = App.create_slider(    self, column=1, row=17, columnspan=2, from_=0.1, to=2, width=155,
                                                    command= lambda value, strvar="lw_var", var="linewidth": self.update_slider_value(value, strvar, var), text="line width", number_of_steps=19, init_val=self.app.linewidth)
        self.alpha_slider       = App.create_slider(    self, column=1, row=18, columnspan=2, from_=0, to=1, width=155,
                                                    command= lambda value, strvar="alpha_var", var="alpha": self.update_slider_value(value, strvar, var), text="line alpha", number_of_steps=20, init_val=self.app.alpha)

        self.reset_button           = App.create_button(self, column=4, row=0, text="Reset Settings",   command=self.reset_values, width=130, pady=(20,5))
        self.minor_ticks_button     = App.create_switch(self, column=4, row=2, text="Use Minor Ticks",  command=lambda: (self.toggle_boolean(self.app.use_minor_ticks),self.apply_settings(None)))
        self.convert_pixels_button  = App.create_switch(self, column=4, row=5, text="Convert Pixels",   command=lambda: self.toggle_boolean(self.app.convert_pixels))
        self.scale_switch_button    = App.create_switch(self, column=4, row=6, text="Use Scalebar",     command=lambda: self.toggle_boolean(self.app.use_scalebar))
        self.cbar_switch_button     = App.create_switch(self, column=4, row=7, text="Use Colorbar",     command=lambda: self.toggle_boolean(self.app.use_colorbar))
        self.convert_rgb_button     = App.create_switch(self, column=4, row=8, text="Convert RGB to gray",command=lambda: self.toggle_boolean(self.app.convert_rgb_to_gray))
        self.save_plain_img_button  = App.create_switch(self, column=4, row=9, text="Save plain images",command=lambda: self.toggle_boolean(self.app.save_plain_image))
        self.hide_params_button     = App.create_switch(self, column=4, row=11, text="Hide fit params",  command=lambda: (self.toggle_boolean(self.app.display_fit_params_in_plot), self.apply_settings(None)))
        self.show_FWHM_button       = App.create_switch(self, column=4, row=12, text="Draw FWHM line", command=lambda: self.toggle_boolean(self.app.draw_FWHM_line))
        self.grid_lines_button      = App.create_switch(self, column=4, row=13, text="Use Grid",         command=lambda: (self.toggle_boolean(self.app.use_grid_lines), self.apply_settings(None)))
        
        self.canvas_width.bind("<KeyRelease>", lambda event: (self.apply_settings(None), self.update_canvas_size(self.app.canvas_ratio)))
        self.canvas_height.bind("<KeyRelease>", lambda event: (self.apply_settings(None), self.update_canvas_size(self.app.canvas_ratio)))

        self.cmap_length_list.configure(state="disabled")
        self.single_colors_list.grid_remove()
        self.canvas_height.grid_remove()
        #tooltips
        self.tooltips = {"canvas_ratio_list":"Determine the size of the canvas. This is useful for saving the plots in a reproducible manner\n- Auto: Resize the canvas to fill the full screen (standard)\n- Custom: Enter the width and height manually using the two entry boxes on the right\n- 4:3 ratio: Set the size of the canvas in 4:3 format, the width [cm] can be set with the entry box on the right",
                    "canvas_width":     "Set the width of the canvas in cm.\n- Works only, if Canvas Size is not set to 'Auto'",
                    "pixel_size":       "Set the size of a single pixel\n- the tick labels can be converted to real lengths by checking 'Convert Pixels'.",
                    "label_dist":       "Set, at which points a new label is placed", 
                    "aspect":           "Set the aspect of the image\n- 'equal' - original image aspect, no pixel distortion\n- 'auto' - fill the image to the size of the window, produces image distortion.",
                    "cmap_imshow_list": "Choose between different colormaps", 
                    "clim_slider":   "Enhance the contrast of the images by selecting the pixel range\n- Uncheck 'Normalize' for manual setting",
                    "plot_type_list":   "Choose between different plotting scales\n- Normal: plt.plot(...)\n- Semi Logarithmic x: plt.semilogx(...)\n- Semi Logarithmic y: plt.semilogy(...)\n- Log-Log plot: plt.loglog(...)",
                    "linestyle_list":   "Choose different line styles for the plot.",
                    "marker_list":      "Determine, if every data point is highlighted with a marker. Different marker shapes can be chosen.", 
                    "cmap_list":        "Choose between different color cycles for multiplots. We differentiate between qualtitative and sequential colormaps:\n- Qualitative Cmap: We cycle through a list of predefined colors of specific length\n- CUSTOM: Use a single color. The color can be chosen by the dropdown menu on the right\n- Sequential Cmap: We create a list of colors based on a continuous colormap. The number of colors can be set by the dropdown menu on the right",
                    "grid_lines_list":  "Check 'Use Grid'\n- Decide between showing major (and minor) grid lines.",
                    "grid_axis_list":  "Check 'Use Grid'\n- Decide between showing only x- oder y-grid lines",
                    "moving_av_list":   "Perform a moving average between neighbouring points to smooth the data.\n- set the number of neighbouring points\n- n = 1: no moving average\n- n=2: left and right neighbouring point",
                    "linewidth_slider": "Choose the line width", 
                    "convert_pixels_button": "Convert the integer pixel labels to lengths in [mm] by stating the 'pixel size'", 
                    "scale_switch_button": "Show a scalebar in the bottom right corner.\n- Scale determined by 'pixel size'\n- Recommendation: use this setting together with 'hide ticks'",
                    "cbar_switch_button": "Show a colorbar on the right to attribute the color to the numeric 'gray-value' between 0 and 1.", 
                    "save_plain_img_button": "When saving a figure with 'Save Figure', only the picture will be saved in the raw format.\nNote: The colormap and current size of the image will be saved.",
                    "hide_params_button": "When using a fit function, the textbox with the fit parameters in the plot will be hidden.",
                    "normalize_list": "Only for line plots:\n- Maximum: Normalize to a maximum value of 1\n- Area: Normalize to area of 1\n- Only Factor: Do not normalize, just multiply the data with the factor", 
                    "normalize_value": "Multiplicative factor to the y-data",
                    "grid_lines_button": "Use Gridlines"
                    }
        
        for name, description in self.tooltips.items():
            setattr(self, name+"_ttp", CreateToolTip(getattr(self, name), description))

        # Slider labels    
        self.pixel_range_var = customtkinter.StringVar()  # StringVar to hold the label value
        self.lw_var = customtkinter.StringVar()  # StringVar to hold the label value
        self.alpha_var = customtkinter.StringVar()  # StringVar to hold the label value
        App.create_label(self, textvariable=self.pixel_range_var, column=2, row=9, width=30, anchor='e', sticky='e')
        App.create_label(self, textvariable=self.lw_var, column=2, row=17, width=30, anchor='e', sticky='e')
        App.create_label(self, textvariable=self.alpha_var, column=2, row=18, width=30, anchor='e', sticky='e')
        self.grid_columnconfigure(3, minsize=30)
        
        self.canvas_width.bind("<KeyRelease>", self.apply_settings)
        self.canvas_height.bind("<KeyRelease>", self.apply_settings)
        self.pixel_size.bind("<KeyRelease>", self.apply_settings)
        self.label_dist.bind("<KeyRelease>", self.apply_settings)
        self.normalize_value.bind("<KeyRelease>", self.apply_settings)

        #set initial values
        self.init_values()
        
    # initiate current values
    def init_values(self):
        # too long to put in the definition of the lists
        self.canvas_ratio_list.set(list(self.canvas_ratio.keys())[list(self.canvas_ratio.values()).index(self.app.canvas_ratio)])
        self.linestyle_list.set(list(self.linestyle.keys())[list(self.linestyle.values()).index(self.app.linestyle)])
        self.marker_list.set(list(self.markers.keys())[list(self.markers.values()).index(self.app.markers)])
        self.plot_type_list.set(list(self.plot_type.keys())[list(self.plot_type.values()).index(self.app.plot_type)])
        self.single_colors_list.set(list(self.single_colors.keys())[list(self.single_colors.values()).index(self.app.single_color)])
        self.update_rangeslider_value(list(self.app.clim), "pixel_range_var")
        self.update_slider_value(self.app.linewidth, "lw_var", "linewidth")
        self.update_slider_value(self.app.alpha, "alpha_var", "alpha")

        if self.app.use_scalebar.get(): self.scale_switch_button.select()
        if self.app.use_colorbar.get(): self.cbar_switch_button.select()
        if self.app.convert_pixels.get(): self.convert_pixels_button.select()
        if not self.app.display_fit_params_in_plot.get(): self.hide_params_button.select()
        if self.app.convert_rgb_to_gray.get(): self.convert_rgb_button.select()
        if self.app.save_plain_image.get(): self.save_plain_img_button.select()
        if self.app.use_grid_lines.get(): self.grid_lines_button.select()
        if self.app.use_minor_ticks.get(): self.minor_ticks_button.select()
        if self.app.draw_FWHM_line.get(): self.show_FWHM_button.select()

    # reset to default values
    def reset_values(self):
        # Here we define the standard values!!!
        self.app.canvas_width = 17
        self.app.pixel_size = 7.4e-3
        self.app.label_dist = 1
        self.app.normalize_value = 1
        self.ticks_settings_list.set(self.ticks_settings[0])
        self.label_settings_list.set(self.label_settings[0])
        self.canvas_width.reinsert(0,str(self.app.canvas_width))
        self.pixel_size.reinsert(0,str(self.app.pixel_size*1e3))
        self.label_dist.reinsert(0,str(self.app.label_dist))
        self.normalize_value.reinsert(0,str(1))
        self.aspect.set(next(iter(self.aspect_values)))
        self.cmap_imshow_list.set(self.cmap_imshow[0])

        self.canvas_ratio_list.set(next(iter(self.canvas_ratio)))
        self.linestyle_list.set(next(iter(self.linestyle)))
        self.marker_list.set(next(iter(self.markers)))
        self.cmap_list.set(self.cmap[0])
        self.single_colors_list.set(next(iter(self.single_colors)))
        self.grid_lines_button.deselect()
        self.grid_lines_list.set(self.grid_lines[0])
        self.grid_axis_list.set(self.grid_axis[0])
        self.moving_av_list.set(str(self.moving_average[0]))
        self.cmap_length_list.set(str(self.cmap_length[1]))
        self.plot_type_list.set(next(iter(self.plot_type)))
        self.normalize_list.set(self.normalize_function[0])

        self.clim_slider.set([0,1])
        self.linewidth_slider.set(1) # before next line!
        self.alpha_slider.set(1) # before next line!
        self.update_rangeslider_value((0,1), "pixel_range_var")
        self.update_slider_value(1, "lw_var", "linewidth")
        self.update_slider_value(1, "alpha_var", "alpha")
          
    def update_slider_value(self, value, strvar_name, var_name):
        variable = getattr(self, strvar_name)
        variable.set(str(round(value,2)))

        setattr(self.app, var_name, value)
        self.app.update_plot(None)

    def update_rangeslider_value(self, value, var_name):
        variable = getattr(self, var_name)
        # (x1, x2) = value
        value = tuple(round(x, 1) for x in value)
        variable.set(f"{value[0],value[1]}")
        self.apply_settings(None)
        
    def apply_settings(self, val):
        try:
            self.app.canvas_width=float(self.canvas_width.get())
            self.app.canvas_height=float(self.canvas_height.get())
            self.app.pixel_size=float(self.pixel_size.get())*1e-3
            self.app.label_dist=float(self.label_dist.get())
        except: pass 
        self.app.normalize_value=float(self.normalize_value.get())
        self.app.aspect=self.aspect.get()
        self.app.cmap_imshow=self.cmap_imshow_list.get()
        self.app.clim=tuple(self.clim_slider.get())

        self.app.label_settings = self.label_settings_list.get()
        self.app.ticks_settings = self.ticks_settings_list.get()
        self.app.linestyle=self.linestyle[self.linestyle_list.get()]
        self.app.markers=self.markers[self.marker_list.get()]
        self.app.cmap=self.cmap_list.get()
        self.app.linewidth=self.linewidth_slider.get()
        self.app.alpha=self.alpha_slider.get()
        self.app.moving_average = int(self.moving_av_list.get())
        self.app.grid_ticks = self.grid_lines_list.get()
        self.app.grid_axis = self.grid_axis_list.get()
        self.app.plot_type=self.plot_type[self.plot_type_list.get()]
        self.app.cmap_length = int(self.cmap_length_list.get())
        self.app.single_color = self.single_colors[self.single_colors_list.get()]
        self.app.normalize_function = self.normalize_list.get()

        if self.cmap_list.get() in sequential_colormaps:
            self.cmap_length_list.configure(state="enabled")
        else: 
            self.cmap_length_list.configure(state="disabled")

        if self.cmap_list.get() == 'CUSTOM':
            self.cmap_length_list.grid_remove()
            self.single_colors_list.grid()
        else: 
            self.single_colors_list.grid_remove()
            self.cmap_length_list.grid()

        if val == None and hasattr(self.app, 'image'): 
            self.app.image.set_clim(self.app.clim)
            if self.app.lineout_button.get():
                self.app.update_lineout_aspect()
            self.app.canvas.draw()
        
        self.app.plot_axis_parameters("ax1", plot_counter=self.app.plot_counter)
        self.app.update_plot(None)

    def toggle_boolean(self, boolean):
        boolean.set(not boolean.get())

    def update_canvas_size(self, canvas_ratio):
        if self.app.canvas_width <= 2 or self.app.canvas_height <= 2: return
        self.app.canvas_ratio = canvas_ratio
        self.app.canvas_widget.pack_forget()

        self.canvas_height.grid() if self.app.canvas_ratio == 0 else self.canvas_height.grid_remove()

        if self.app.canvas_ratio is not None:
            width = self.app.canvas_width/2.54
            height = self.app.canvas_height/2.54 if self.app.canvas_ratio == 0 else self.app.canvas_width/(self.app.canvas_ratio*2.54)
            self.app.fig.set_size_inches(width, height)
            self.app.canvas_widget.pack(expand=True, fill=None) 
            self.app.canvas_widget.config(width=self.app.fig.get_size_inches()[0] * self.app.fig.dpi, height=self.app.fig.get_size_inches()[1] * self.app.fig.dpi)
        else:
            width = (self.app.tabview.winfo_width() - 18) / self.app.fig.dpi
            height = (self.app.tabview.winfo_height()) / self.app.fig.dpi
            self.app.fig.set_size_inches(w=width, h=height, forward=True)  # Re-enable dynamic resizing
            self.app.canvas_widget.pack(fill="both", expand=True) 

        self.app.canvas.draw()  # Redraw canvas to apply the automatic size

"""
############################################################################################
##############################     Legend Window     #######################################
############################################################################################

"""

class LegendWindow(customtkinter.CTkToplevel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry("600x250")
        self.title("Legend Settings")
        self.app = app
        self.legend_type = {'Best location': {'loc': 'best'}, 
                            'Upper right': {'loc': 'upper right'}, 
                            'Upper left': {'loc': 'upper left'},
                            'Lower left': {'loc': 'lower left'},
                            'Lower right': {'loc': 'lower right'}, 
                            'Outer right ↑': {'loc': 'upper left', 'bbox_to_anchor': (1,1)}, 
                            'Outer right ↓': {'loc': 'lower left', 'bbox_to_anchor': (1,0)}
                            }
        
        self.legend_name = {'Custom name': slice(None,None),
                            'Full name': slice(0,None), 
                            'Remove file type': slice(0,-4),
                            }

        self.grid_columnconfigure(0, minsize=120)
        self.grid_columnconfigure(3, minsize=30)
        App.create_label(self, column=1, row=0, columnspan=2, text="Legend Settings", font=customtkinter.CTkFont(size=16, weight="bold"),padx=20, pady=(20, 5), sticky=None)

        # values for line plots
        self.reset_button       = App.create_button(self, column=4, row=1, text="Reset Settings",   command=self.reset_values, width=130)
        self.legend_type_list   = App.create_Menu(self, column=1, row=1, columnspan=2, values=list(self.legend_type.keys()), text="Legend location", command=self.apply_settings)
        self.legend_name_list   = App.create_Menu(self, column=1, row=2, columnspan=2, values=list(self.legend_name.keys()), text="Legend: {name}", command=self.apply_settings)
        self.slice_var = customtkinter.StringVar()  # StringVar to hold the label value
        self.fontsize_var = customtkinter.StringVar()  # StringVar to hold the label value
        App.create_label(self, textvariable=self.slice_var, column=1, row=5, width=50, columnspan=3, sticky="w", padx=10)
        App.create_label(self, textvariable=self.fontsize_var, column=2, row=3, width=30, sticky="e")

        self.name_slice_slider  = App.create_range_slider(self, from_=0, to=len(self.app.optmenu.get()), text="slice file name", command= lambda value, var="slice_var": self.update_slider_value(value, var), row=4, column =1, width=200, columnspan=2, init_value=[0, len(self.app.optmenu.get())])
        self.fontsize_slider   = App.create_slider(    self, column=1, row=3, columnspan=2, from_=5, to=20, 
                                                    command= lambda value, var="fontsize_var": self.update_fontsize_value(value, var), text="font size", width=160, number_of_steps=15, init_val=self.app.legend_font_size)

        self.update_labels_button  = App.create_switch(self, column=4, row=2, text="Update labels",   command=lambda: self.toggle_boolean(self.app.update_labels_multiplot))
        self.show_title_entry_button  = App.create_switch(self, column=4, row=3, text="Show title entry",   command=lambda: (self.toggle_boolean(self.app.show_title_entry), self.app.use_labels()))
        #set initial values
        self.init_values()
        
    # initiate current values
    def init_values(self):
        self.legend_type_list.set(list(self.legend_type.keys())[list(self.legend_type.values()).index(self.app.legend_type)])
        try:
            self.legend_name_list.set(list(self.legend_name.keys())[list(self.legend_name.values()).index(self.app.legend_name)])
        except ValueError:
            self.legend_name_list.set(list(self.legend_name.keys())[list(self.legend_name.values()).index(slice(None,None))])
        self.slice_var.set(self.app.optmenu.get())
        self.fontsize_var.set(str(round(self.app.legend_font_size)))
        if self.app.update_labels_multiplot.get(): self.update_labels_button.select()
        if self.app.show_title_entry.get(): self.show_title_entry_button.select()

    # reset to defaul values
    def reset_values(self):
        self.legend_type_list.set(next(iter(self.legend_type)))
        self.legend_name_list.set(next(iter(self.legend_name)))
        self.fontsize_slider.set(10)
        self.fontsize_var.set(str(10))
        
        self.apply_settings(None)
        # Here we define the standard values!!!
         
    def update_slider_value(self, value, var_name):
        variable = getattr(self, var_name)
        (x1, x2) = value 
        slice_object = slice(int(x1), int(x2))
        variable.set(self.app.optmenu.get()[slice_object])
        self.legend_name['Custom name'] = slice_object
        self.apply_settings(None)

    def update_fontsize_value(self, value, var_name):
        variable = getattr(self, var_name)
        variable.set(str(round(value,2)))
        self.apply_settings(None)
        
    def apply_settings(self, val):
        # self.app.aspect=self.aspect.get()
        self.app.legend_type = self.legend_type[self.legend_type_list.get()]
        self.app.legend_name = self.legend_name[self.legend_name_list.get()]
        self.app.legend_font_size = float(self.fontsize_slider.get())

    def toggle_boolean(self, boolean):
        boolean.set(not boolean.get())

'''
############################################################################################
##############################    Fitting Window     #######################################
############################################################################################
'''         
# neues Fenster
class FitWindow(customtkinter.CTkFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.app = app
        self.row = 20
        self.widget_dict = {}
        self.fitted_label_string = ""
        self.labels_title = App.create_label(app.settings_frame, text="Fit Function", font=customtkinter.CTkFont(size=16, weight="bold"),row=self.row , column=0, columnspan=5, padx=20, pady=(20, 5),sticky=None)
        self.function = {'Gaussian': gauss, 'Gaussian 3rd': gauss3, 'Lorentz': lorentz, 'Linear': linear, 'Quadratic': quadratic, 'Exponential': exponential, 'Logarithmic': logarithm ,'Square Root': sqrt, 'Hyperbola': hyperbola}

        self.function_label = App.create_label(app.settings_frame, text="", column=1, row=self.row +1, width=80, columnspan=4, anchor='e', sticky="w", pady=(0,10))
        self.function_label_list = {'Gaussian': ["f(x) = a·exp(-(x-b)²/(2·c²)) + d", r"f(x) = a \cdot \exp\left(-\,\dfrac{(x-b)^2}{(2·c^2)}\right) + d"], 
                                    'Gaussian 3rd': ["f(x) = a·exp(-|x-b|³/(2·c²)) + d", r"f(x) = a \cdot \exp\left(-\,\dfrac{|x-b|^3}{(2·c^2)}\right) + d"], 
                                    'Lorentz': ["f(x) = a²/[(x²-b²)² + c²·b²]+d", r"f(x) = \dfrac{a^2}{(x^2 - b^2)^2 + c^2 \cdot b^2}+d"],
                                    'Linear': ["f(x) = a·x + b", r"f(x) = a\cdot x + b"], 
                                    'Quadratic': ["f(x) = a·x² + b·x + c", r"f(x) = a\cdot x^2 + b\cdot x+c"], 
                                    'Exponential': ["f(x) = a·exp(b·x) + c", r"f(x) = a\cdot \exp(b\cdot x) +c"], 
                                    'Logarithmic': ["f(x) = a·ln(b·x) + c", r"f(x) = a\cdot \ln(b\cdot x) + c"],
                                    'Square Root': ["f(x) = a·sqrt(b·x) + c", r"f(x) = a\sqrt{b\cdot x} + c"],
                                    'Hyperbola': ["f(x) = a/(x-b) + c", r"f(x) = \dfrac{a}{x-b} + c"]
                                    }
        self.function_list_label = App.create_label(app.settings_frame,text="Fit", column=0,row=self.row +2, sticky="e")
        self.function_list = App.create_Menu(app.settings_frame, values=list(self.function.keys()), command=self.create_params, width=110, column=1, row=self.row +2, columnspan=2, sticky="w")
        self.function_list.set('Gaussian')

        if app.lineout_button.get():
            self.data = app.lineout_data
        else:
            self.data = app.data

        self.save_fit_button = App.create_button(app.settings_frame, column = 3, row=self.row+2, text="Save fit", command= lambda: app.save_data_file(np.vstack([self.data[:,0], app.fit_plot(app.function, app.params, self.data)]).T), width=110, columnspan=2)
        if (not app.image_plot or app.lineout_button.get()):
            self.fit_borders_slider = App.create_range_slider(app.settings_frame, from_=np.min(self.data[:,0]), to=np.max(self.data[:,0]), command= lambda val=None: app.update_plot(val), row=self.row+3, column =1, width=180, padx=(10,10), columnspan=4, init_value=[np.min(self.data[:,0]), np.max(self.data[:,0])])
            self.fit_borders_label = App.create_label(app.settings_frame, row=self.row+3, column=0, text="borders")

        self.params = []
        self.error = []

        for i,name in enumerate(["a","b","c","d"]):
            self.create_widget(name, i)

        self.widget_dict['str_var_FWHM'] = customtkinter.StringVar()
        self.create_params('Gaussian')
        app.settings_frame.grid()
    
    # Function to render LaTeX using Matplotlib
    def render_latex_to_image(self, latex_code):
        fig, ax = plt.subplots()
        text = ax.text(0.5, 0.5, f'${latex_code}$', fontsize=10, ha='center', va='center', color=app.text_color)
        ax.axis('off')

        # Draw the canvas to update the text position
        fig.canvas.draw()
        # Get the bounding box of the text
        bbox = text.get_window_extent()
        # Convert bbox to display coordinates
        bbox_display = bbox.transformed(fig.dpi_scale_trans.inverted())
        # Get the width and height of the bounding box
        width, height = bbox_display.width, bbox_display.height
        # Adjust figure size to fit the text exactly
        fig.set_size_inches(width, height)
        
        buf = io.BytesIO()
        # enable transparent background for creating images of equations
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
        buf.seek(0)
        
        image = Image.open(buf)
        plt.close(fig)
        
        return image

    # Show/Hide the fit parameter widgets and display the function name with LaTeX
    def create_params(self, function_name):
        # set the label of the function, e.g f(x) = a*x+b
        self.function_name = function_name
        latex_code = self.function_label_list[self.function_list.get()][1]
        image = self.render_latex_to_image(latex_code)
        self.function_label.configure(image=customtkinter.CTkImage(dark_image=image, size=(image.width, image.height)))
        
        #Determine the number of arguments of the used function
        self.number_of_args = len(signature(self.function[function_name]).parameters) - 1
        
        for i,name in enumerate(["a","b","c","d"]):
            if i < self.number_of_args:
                [self.widget_dict[f'{attribute}_{name}'].grid() for attribute in ["fit", "fitlabel", "fitted"]]
            else:
                [self.widget_dict[f'{attribute}_{name}'].grid_remove() for attribute in ["fit", "fitlabel", "fitted"]]

        self.set_fit_params(self.data)
        self.row += 2 + self.number_of_args
    
    # Create Fit parameter widget, its name label widget and a string Variable to hold the label vlue
    def create_widget(self, name, arg_number):
        self.widget_dict[f'fit_{name}'], self.widget_dict[f'fitlabel_{name}'] = App.create_entry(app.settings_frame, column=1, row=self.row + arg_number + 4, width=80, placeholder_text="0", sticky='w', columnspan=2, text=name, textwidget=True)
        self.widget_dict[f'str_var_{name}'] = customtkinter.StringVar() # StringVar to hold the label value
        self.widget_dict[f'fitted_{name}'] = App.create_label(app.settings_frame, textvariable=self.widget_dict[f'str_var_{name}'], column=3, width=50, row=self.row + arg_number + 4, padx=(0, 20), columnspan=2)
    
    # Write the fitted parameter values in the labels
    def set_fitted_values(self, params, error):
        self.fitted_label_string = self.function_label_list[self.function_list.get()][0] 
        for i,name in enumerate(["a","b","c","d","FWHM"]):
            if self.function_name == "Gaussian":
                self.number_of_args += 1
                params = np.append(params, 2*np.sqrt(2*np.log(2))*params[2]) 
                error = np.pad(error, ((0, 1), (0, 1)), mode='constant')
                error[-1,-1] = error[2,2] * params[-1]
            
            if i >= self.number_of_args: break   # take only the right number of arguments
            self.params = params
            self.error = error
            

            try:
                round_digit = -int(np.floor(np.log10(abs(np.sqrt(error[i,i])))))
            except:
                round_digit = 4
            self.widget_dict[f'str_var_{name}'].set(str(round(params[i],round_digit))+" ± "+str(round(np.sqrt(error[i,i]),round_digit)))
            self.fitted_label_string += "\n" + name + " = " + self.widget_dict[f'str_var_{name}'].get()
    
    # display values in the textbox, function is used in app.plot()
    def get_fitted_labels(self):
        return self.fitted_label_string

    # Set the initial guess parameters    
    def set_fit_params(self, data):
        self.app.params = []
        for i,name in enumerate(['a','b','c','d']):
            if i >= self.number_of_args: break   # take only the right number of arguments
            try:
                value = float(self.widget_dict[f'fit_{name}'].get())
                self.app.params.append(value)
            except ValueError:
                self.app.params.append(1)
                if (self.function_name in ["Gaussian","Gaussian 3rd"]) and i == 1:
                    self.app.params[1] = data[np.argmax(data[:,1]),0]
                    self.app.params[0] = np.max(data[:,1])
                if (self.function_name in ["Lorentz"]) and i == 1:
                    self.app.params[1] = data[np.argmax(data[:,1]),0]
                    self.app.params[0] = np.sqrt(np.max(data[:,1]))*data[np.argmax(data[:,1]),0]

        self.app.function = self.function[self.function_list.get()]
        self.app.use_fit = app.fit_button.get()
        
    def close_window(self):
        for name in ["a","b","c","d"]:
            [self.widget_dict[f'{attribute}_{name}'].grid_remove() for attribute in ["fit", "fitlabel", "fitted"]]
    
        widget_names = ["labels_title", "function_list", "function_list_label", "function_label", "fit_borders_slider", "fit_borders_label", "save_fit_button"]
        [getattr(self, name).grid_remove() for name in widget_names]
        self.app.use_fit = app.fit_button.get()

'''
############################################################################################
##############################     Table  Window     #######################################
############################################################################################
'''
# class Table(customtkinter.CTkScrollableFrame):
#     def __init__(self, master, data, **kwargs):
#         super().__init__(master, **kwargs)

#         self.columns = ("x", "y")
#         self.data = data

#         self.text_widget = customtkinter.CTkTextbox(self, padx=10, pady=5)
#         self.text_widget.grid(row=0, column=0, sticky="nsew")
#         self.grid_columnconfigure(0, weight=1)
        
#         self.text_widget.insert("1.0", "\t".join(self.columns) + "\n")

#         for row in self.data:
#             # self.insert_data(row)
#             self.text_widget.insert("end", "\t".join(map(str, row)) + "\n")

class CreateToolTip(object):
    """
    https://stackoverflow.com/questions/3221956/how-do-i-display-tooltips-in-tkinter
    create a tooltip for a given widget
    """
    def __init__(self, widget, text='widget info'):
        self.waittime = 1000     #miliseconds
        self.wraplength = 300   #pixels
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.widget.bind("<ButtonPress>", self.leave)
        self.id = None
        self.tw = None

    def enter(self, event=None):
        if tooltips_enabled:
            self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.waittime, self.showtip)

    def unschedule(self):
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)

    def showtip(self, event=None):
        x = y = 0
        bbox = self.widget.bbox("insert")

        if bbox:
            x, y, cx, cy = self.widget.bbox("insert")
            # Adjust the tooltip width based on DPI scaling
            wraplength = int(self.widget.winfo_fpixels(f'{self.wraplength}p'))
            screen_width = self.widget.winfo_screenwidth()

            x += self.widget.winfo_rootx() + 0
            y += self.widget.winfo_rooty() + 40

            if x + wraplength > screen_width:
                x = screen_width - wraplength - 120  # Adjust position to avoid overflow

            self.hidetip()
            # creates a toplevel window
            self.tw = customtkinter.CTkToplevel(self.widget, fg_color = ["#f9f9fa","#343638"]) #["#979da2","#565b5e"]
            # Leaves only the label and removes the app window
            self.tw.wm_overrideredirect(True)
            self.tw.wm_geometry("+%d+%d" % (x, y))
            label = customtkinter.CTkLabel(self.tw, text=self.text + "\nPress F1 to deactivate tool tips", justify='left', wraplength = wraplength, fg_color = "transparent", padx=10, pady=5)
            label.pack()
        else: 
            logging.error("The cursor of the text widget is invisible and the bounding box cannot be determined")

    def hidetip(self):
        tw = self.tw
        self.tw= None
        if tw:
            tw.destroy()
    
    def cleanup(self):
        self.unschedule()
        self.hidetip()


class CustomToolbar(NavigationToolbar2Tk):
    # Modify the toolitems list to remove specific buttons
    def set_message(self, s):
        formatted_message = s.replace("\n", ", ").strip()
        self.message.set(formatted_message)

    toolitems = (
        ('Home', 'Reset original view', 'home', 'home'),
        ('Back', 'Back to previous view', 'back', 'back'),
        ('Forward', 'Forward to next view', 'forward', 'forward'),
        ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
        ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
        # Add or remove toolitems as needed
        # ('Subplots', 'Configure subplots', 'subplots', 'configure_subplots'),
        # ('Save', 'Save the figure', 'filesave', 'save_figure'),
    )

# Extend the class of the entry widget to add the reinsert method
class CustomEntry(customtkinter.CTkEntry):
    def reinsert(self, index, text):
        self.delete(0, 'end')  # Delete the current text
        self.insert(index, text)  # Insert the new text

if __name__ == "__main__":
    level = logging.INFO # INFO
    fmt = '[%(levelname)s] %(asctime)s - %(message)s'
    logging.basicConfig(level=level, format=fmt)

    app = App()
    app.state('zoomed')
    app.protocol("WM_DELETE_WINDOW", app.on_closing)

    app.bind("<Control-z>", lambda x: app.control_z())
    app.bind("<Control-o>", lambda x: app.read_file_list())
    app.bind("<Control-s>", lambda x: app.save_figure())
    app.bind("<Control-n>", lambda x: app.normalize_button.toggle())
    app.bind("<Control-r>", lambda x: app.initialize())
    app.bind("<Control-m>", lambda x: app.plot())
    app.bind("<Control-Left>", lambda x: app.change_plot(app.replot,-1))
    app.bind("<Control-Right>", lambda x: app.change_plot(app.replot,1))
    app.bind("<F1>", toggle_bool)
    app.mainloop()
