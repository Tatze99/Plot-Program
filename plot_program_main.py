# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 14:28:26 2022

@author: Martin

to do: implement minor ticks as option
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
# import io
from pandas import read_table
import matplotlib
from cycler import cycler
from scipy.optimize import curve_fit as cf
from scipy.interpolate import interp1d
from inspect import signature # get number of arguments of a function
import cv2 # image fourier transform
from PIL import Image, ImageTk
from sys import exit as sysexit
from natsort import natsorted
import io

myappid = 'mycompany.myproduct.subproduct.version' # arbitrary string
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

tooltips_enabled = True

def toggle_bool(event=None):
    global tooltips_enabled
    tooltips_enabled = not tooltips_enabled

version_number = "24/08"
plt.style.use('default')
matplotlib.rc('font', family='serif')
matplotlib.rc('font', serif='Times New Roman')
file_type_names = ('.csv', '.dat', '.txt', '.png', '.jpg', '.jpeg', '.spec', '.JPG', '.bmp', '.webp', '.tif', '.tiff', '.PNG', '.pgm', '.pbm', '.lvm')
image_type_names = ('png','.jpg', '.jpeg', '.JPG', '.bmp', '.webp', '.tif', '.tiff', '.PNG', '.pgm', '.pbm')
sequential_colormaps = ['magma','hot','viridis', 'plasma', 'inferno', 'cividis', 'gray', 'bone', 'afmhot', 'copper','Purples', 'Blues', 'Greens', 'Oranges', 'Reds','twilight', 'hsv', 'rainbow', 'jet', 'turbo', 'gnuplot', 'brg']


ctypes.windll.shcore.SetProcessDpiAwareness(1)

Standard_path = os.path.dirname(os.path.abspath(__file__))
filelist = natsorted([fname for fname in os.listdir(Standard_path) if fname.endswith(file_type_names)])

def exponential(x, a, b, c):
    return a * np.exp(b * x) + c

def lorentz(x, a, w0, gamma):
    return a / ((x**2 - w0**2)**2 + gamma**2*w0**2)

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

    elif (theta == 90 or theta == -90):
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

    return round(x1), round(x2), round(y1), round(y2)

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
        self.plot_index = 1
        self.color = "#1a1a1a" # toolbar
        self.text_color = "white"
        self.reset_plots = True
        # self.reset_xlims = customtkinter.BooleanVar(value=False)
        # self.reset_ylims = customtkinter.BooleanVar(value=False)
        self.legend_type = {'loc': 'best'}
        self.legend_name = None
        self.canvas_ratio = None #None - choose ratio automatically
        self.canvas_width = 17 #cm
        self.label_settings = "smart"
        self.ticks_settings = "smart"
        self.multiplot_row = 9

        # line plot settings
        self.linestyle='-'
        self.markers=''
        self.cmap="tab10"
        self.linewidth=1
        self.moving_average = 1
        self.change_norm_to_area = customtkinter.BooleanVar(value=False)
        self.hide_ticks = customtkinter.BooleanVar(value=False)
        self.use_grid_lines = customtkinter.BooleanVar(value=False)
        self.grid_ticks = 'major'
        self.grid_axis = 'both'
        self.cmap_length = 10
        self.single_color = 'tab:blue'
        self.sub_plot_counter = 1
        self.sub_plot_value = 1
        self.first_ax2 = True
        
        # image plot settings
        self.cmap_imshow="magma"
        self.interpolation = None
        self.enhance_value = 1 
        self.pixel_size = 7.4e-3 # mm
        self.label_dist = 1
        self.aspect = "equal"
        self.plot_type = "plot"

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
        
        self.tabview = customtkinter.CTkTabview(self, width=250)
        self.tabview.grid(row=3, column=1, padx=(10, 10), pady=(20, 0), columnspan=2, sticky="nsew", rowspan=10)
        self.tabview.add("Show Plots")
        self.tabview.add("Data Table")
        # self.tabview.tab("Show Plots").configure(fg_color="white")
        # self.tabview.rowconfigure(0, weight=1)  # Make the plot area expandable vertically
        # self.tabview.columnconfigure(0, weight=1) 
        
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
        self.normalize_button = App.create_switch(frame, text="Normalize",  command=lambda: self.plot,    column=0, row=11, padx=20)
        self.lineout_button   = App.create_switch(frame, text="Lineout",  command=self.lineout,    column=0, row=12, padx=20)
        self.FFT_button       = App.create_switch(frame, text="FFT",  command=self.fourier_trafo,    column=0, row=13, padx=20)
        # self.Advanced_button  = App.create_switch(frame, text="Advanced Settings",  command=self.advanced_settings,    column=0, row=16, padx=20)
        self.data_table_button= App.create_switch(self.tabview.tab("Data Table"), text="Plot data table & custom data", command=None, row=0, column=0, padx=20)
        self.mid_axis_button  = App.create_switch(self.tabview.tab("Data Table"), text="middle axis lines", command=None, row=1, column=0, padx=20)
        self.data_table = App.create_table(self.tabview.tab("Data Table"), data=np.vstack((np.linspace(0,5,100),np.linspace(0,5,100)**2)).T, width=300, sticky='ns', row=2, column=0, pady=(20,10), padx=10)
        self.xval_min_entry, self.xlims_entry_label      = App.create_entry(self.tabview.tab("Data Table"), row=0, column=3, width=90, text="x-limits", placeholder_text="x min")
        self.xval_max_entry                              = App.create_entry(self.tabview.tab("Data Table"), row=0, column=4, width=90, placeholder_text="x max")
        self.function_entry, self.function_entry_label   = App.create_entry(self.tabview.tab("Data Table"), row=1, column=3, width=200, text="y-Function", columnspan=2, placeholder_text="np.sin(x)", sticky="nw", sticky_label="ne")
        
        #tooltips
        self.tooltips = {"load_button": "Load a folder with data files\n- only files in this directory will be displayed in the list (no subfolders)\n- Standard path: Location of the Script or the .exe file",
                    "save_button": "Save the current image or current data\n- Image file types: .png, .jpg, .pdf\n- Data file types: .dat, .txt, .csv\n- Images can be saved in a plain format (without the axis) by checking 'Plot settings' -> 'Save plain images'\n- The image format can be chosen via 'Plot settings' -> 'Canvas Size'. Make sure to maximize the application window, otherwise the ratio can be altered by a large sidebar",
                    "set_button":       "Open the Plot settings window\n- Top: Image Plot settings\n- Bottom Line Plot settings\n- Side: Boolean settings",
                    "plot_button":      "Reset and reinitialize the plot \n- delete all previous multiplots",
                    "reset_button":     "Reset and reinitialize the plot \n- delete all previous multiplots",
                    "addplot_button":   "Plot the currently selected file",
                    "prev_button":      "Plot the previous file in the list",
                    "next_button":      "Plot the next file in the list",
                    "multiplot_button": "Create multiple plots in one figure \n- choose rows = 1, cols = 1 for single graph multiplots\n- Press CTRL+Z to delete the previous plot\n- You can click on the subplots to replot them or set the limits",
                    "uselabels_button": "Create x,y-labels, add a legend\n- Legend settings: choose location, add file name to legend",
                    "uselims_button":   "Choose x,y-limits for line and image plots\n- x-limits stay the same when resetting the plot\n- y-limits will change to display all figures\n- the limits can be set with the sliders or typed in manually", 
                    "fit_button":       "Fit a function to the current data set\n- initial fit parameters can be set when the fit does not converge\n- use slider to fit only parts of the function\n- the textbox can be hidden via the 'Plot settings' -> 'Hide fit params'",
                    "normalize_button": "Normalize the plot\n- data file: normalize to max = 1 or area = 1 (Plot settings -> normalize Area)\n- image file: enhance the contrast of the image (can be changed by adjusting: Plot settings -> Enhance value)", 
                    "lineout_button": "Display a lineout function\n- only used for images, no multiplot possible", 
                    "FFT_button": "Display the |FT(f)|² of the data\n- limits of the Fourier window can be set\n- zero padding (factor determines number of zeros/x-array length)",
                    "data_table_button": "Check this to plot own custom data in the window below\n- columns must be separated by a space or tab (not comma or semicolon)\n- decimal separator: point and comma works\n- You can also plot data by using numpy functions, the argument must be called 'x'."
                    }
        
        for name, description in self.tooltips.items():
            setattr(self, name+"_ttp", CreateToolTip(getattr(self, name), description))

        for name in ["multiplot_button", "uselabels_button", "uselims_button", "fit_button", "normalize_button", "lineout_button", "FFT_button"]:
                getattr(self, name).configure(state="disabled")

        self.settings_frame = customtkinter.CTkFrame(self, width=1, height=600, corner_radius=0)
        self.settings_frame.grid(row=0, column=4, rowspan=999, sticky="nesw")
        self.columnconfigure(2,weight=1)
        self.two_axis_button = App.create_switch(self.settings_frame, command=None, text="2nd axis", column=2, row=self.multiplot_row+2, columnspan=2, padx=20, pady=(10,5))
        self.two_axis_button.grid_remove()
        # self.columnconfigure(1,weight=1)
        
        self.appearance_mode_label = App.create_label(frame, text="Appearance Mode:", row=17, column=0, padx=20, pady=(10, 0), sticky="w")
        self.appearance_mode_optionemenu = App.create_optionMenu(frame, values=["Dark","Light", "System"], command=self.change_appearance_mode_event, width=200, row=18, column=0, padx=20, pady=(5,10))
        
        #entries
        self.folder_path, self.folder_path_label = self.create_entry(column=2, row=0, text="Folder path", columnspan=2, width=600, padx=10, pady=10, sticky="w")
        
        #dropdown menu
        self.optmenu = self.create_combobox(values=filelist, text="File name",row=1, column=2, columnspan=2, width=600, sticky="w")
        if not filelist == []: self.optmenu.set(filelist[0])

    def create_label(self, row, column, width=20, text=None, anchor='e', sticky='e', textvariable=None, padx=(5,5), image=None, pady=None, font=None, columnspan=1, fg_color=None, **kwargs):
        label = customtkinter.CTkLabel(self, text=text, textvariable=textvariable, width=width, image=image, anchor=anchor, font=font, fg_color=fg_color,  **kwargs)
        label.grid(row=row, column=column, sticky=sticky, columnspan=columnspan, padx=padx, pady=pady, **kwargs)
        return label

    def create_entry(self, row, column, width=200, text=None, columnspan=1, padx=10, pady=5, placeholder_text=None, sticky='w', sticky_label='e', **kwargs):
        entry = customtkinter.CTkEntry(self, width, placeholder_text=placeholder_text, **kwargs)
        entry.grid(row=row, column=column, columnspan=columnspan, padx=padx, pady=pady, sticky=sticky, **kwargs)
        if text is not None:
            entry_label = App.create_label(self, text=text, column=column-1, row=row, width=20, anchor='e', sticky=sticky_label)
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

    #240202 - text -> text=None
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
    
    def create_optionMenu(self, values, column, row, command=None, text=None, width=200, columnspan=1, padx=10, pady=5, sticky=None, **kwargs):
        combobox = customtkinter.CTkOptionMenu(self, values=values, width=width, command=command, **kwargs)
        combobox.grid(column=column, row=row, columnspan=columnspan, padx=padx, pady=pady, sticky=sticky, **kwargs)
        if text is not None:
            App.create_label(self, text=text, column=column-1, row=row, anchor='e', pady=pady)
        return combobox
    
    def create_slider(self, from_, to, row, column, width=200, text=None, init_value=None, command=None, columnspan=1, number_of_steps=1000, padx = 10, pady=5, sticky='w',**kwargs):
        slider = customtkinter.CTkSlider(self, from_=from_, to=to, width=width, command=command, number_of_steps=number_of_steps)
        slider.grid(column=column, row=row, columnspan=columnspan, padx=padx, pady=pady, sticky=sticky, **kwargs)
        if text is not None:
            App.create_label(self, text=text, column=column-1, row=row, width=20, anchor='e')

        if init_value is not None:
            slider.set(init_value)

        return slider
    
    def create_range_slider(self, from_, to, row, column, width=200, text=None, init_value=None, command=None, columnspan=1, number_of_steps=1000, padx = 10, pady=5, sticky='w',**kwargs):
        slider = CTkRangeSlider(self, from_=from_, to=to, width=width, command=command, number_of_steps=number_of_steps)
        slider.grid(column=column, row=row, columnspan=columnspan, padx=padx, pady=pady, sticky=sticky, **kwargs)
        if text is not None:
            App.create_label(self, text=text, column=column-1, row=row, width=20, anchor='e')

        if init_value is not None:
            slider.set(init_value)

        return slider
    
    def create_table(self, data,  width, row, column,header=None, sticky=None, **kwargs):
        text_widget = customtkinter.CTkTextbox(self, width = width, padx=10, pady=5)
        # text_widget.pack(fill="y", expand=True)
        text_widget.grid(row=row, column=column, sticky=sticky, **kwargs)
        self.grid_rowconfigure(row, weight=1)
        if header is not None:
            text_widget.insert("0.0", "\t".join(header) + "\n")

        for row in data:
            text_widget.insert("end", " \t ".join(map(str, np.round(row,2))) + "\n")
        return text_widget
    
    #############################################################
    ######################## Do the plots #######################
    #############################################################

    def initialize(self):
        if not self.initialize_plot_has_been_called:
            self.initialize_plot_has_been_called = True
            for name in ["multiplot_button", "uselabels_button", "uselims_button", "fit_button", "normalize_button", "lineout_button", "FFT_button"]:
                getattr(self, name).configure(state="enabled")

        if self.lineout_button.get():
            self.initialize_lineout(self.choose_ax_button.get())
        elif self.FFT_button.get():
            self.initialize_FFT()
        else:
            self.initialize_plot()

    def initialize_plot(self):
        
        # Clear the previous plot content
        if self.ax1 is not None:
            if self.ax2 is not None: self.ax2.clear()
            self.ax2 = None
            self.first_ax2 = True
            self.ax1.clear()  
            self.canvas.get_tk_widget().destroy()
            self.toolbar.destroy()
            self.plot_counter = 1      # counts how many plots there are
            self.plot_index = 1        # index of the currently used plot
            self.sub_plot_counter = 1  # number of subplots
            self.mouse_clicked_on_canvas = False
            plt.close(self.fig)
            
        self.ax_container = []
            
        if self.replot and not self.lineout_button.get() and not self.FFT_button.get():
            self.rows=float(self.ent_rows.get())
            self.cols=float(self.ent_cols.get())
            self.sub_plot_value = float(self.ent_subplot.get())
        
        # self.canvas.get_tk_widget().grid(row=2, column=2, columnspan=2)
        if self.canvas_ratio == None:
            self.fig = plt.figure(constrained_layout=True, dpi=150)  
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.tabview.tab("Show Plots"))
            self.canvas.get_tk_widget().pack(fill="both", expand=True)
            self.canvas.mpl_connect("button_press_event", self.on_click)
        else:
            self.fig = plt.figure(figsize=(self.canvas_width/2.54,self.canvas_width/(self.canvas_ratio*2.54)),constrained_layout=True, dpi=150)  
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.tabview.tab("Show Plots"))
            self.canvas.get_tk_widget().pack(expand=True)

        self.toolbar = self.create_toolbar()
        self.ymax = float('-inf')
        self.ymin = float('inf')
        self.xmax = float('-inf')
        self.xmin = float('inf')

        self.plot()
        if ((self.uselims_button.get() and self.reset_plots and not self.lineout_button.get() and not self.FFT_button.get()) or (self.uselims_button.get() and self.image_plot != self.image_plot_prev)): 
            # make sure the sliders are
            self.update_slider_limits()
            xlim_min, xlim_max, ylim_min, ylim_max = self.reset_limits()

            # if self.reset_xlims.get():
            # if self.reset_ylims.get():
            if self.image_plot != self.image_plot_prev:
                self.xlim_slider.set([xlim_min, xlim_max])
                self.ylim_slider.set([ylim_min, ylim_max])
            
            self.update_plot("Update Limits")
        
    def plot(self):
        file_path = os.path.join(self.folder_path.get(), self.optmenu.get())

        # Decide if there is an image to process or a data file
        self.image_plot_prev = self.image_plot
        if file_path.endswith(image_type_names) and not self.data_table_button.get():
            self.image_plot = True 
        else:
            self.image_plot = False
        
        if self.fit_button.get() and self.image_plot and not self.lineout_button.get(): self.fit_button.toggle()
        if self.lineout_button.get() and not self.image_plot: self.lineout_button.toggle()
        if not self.multiplot_button.get() and self.ax1 is not None: self.ax1.remove()   # reset the plot for normalize but use it normally in multiplot

        # multiplots but in several subplots
        if self.replot and (self.rows != 1 or self.cols != 1):
            if self.sub_plot_counter >= self.sub_plot_value or self.plot_counter == 1:
                if self.plot_counter > self.rows*self.cols: return
                ax1 = self.fig.add_subplot(int(self.rows),int(self.cols),self.plot_counter)
                if self.mouse_clicked_on_canvas:
                    self.ax1.remove()
                    self.ax_container[self.plot_counter-1] = (ax1, None)
                    self.mouse_clicked_on_canvas = False
                else: 
                    self.ax_container.append((ax1, None)) # Add primary axis with no twin yet
                self.ax1 = ax1
                self.ax1.set_prop_cycle(cycler(color=self.get_colormap(self.cmap)))
                self.sub_plot_counter = 1
                self.first_ax2 = True
                self.two_axis_button.deselect()
                self.plot_index = self.plot_counter
            else: 
                self.plot_counter -= 1
                self.sub_plot_counter += 1
        # single plot or multiplot but same axis, no image plot!
        elif (self.replot and (self.rows == 1 and self.cols == 1) and self.plot_counter == 1) or not self.replot:
            self.ax1 = self.fig.add_subplot(1,1,1)
            self.ax1.set_prop_cycle(cycler(color=self.get_colormap(self.cmap)))

        # create the plot depending if we have a line or image plot
        if self.image_plot: self.make_image_plot(file_path)
        else:
            file_decimal = self.open_file(file_path)
            if not self.data_table_button.get():
                try:
                    self.data = np.array(read_table(file_path, decimal=file_decimal, skiprows=self.skip_rows(file_path), skip_blank_lines=True, dtype=np.float64))
                    print("plot with pandas")
                except: 
                    self.data = np.loadtxt(file_path)
                    print("plot with np.loadtxt")
            else: 
                self.data = []
    
            line_container = self.make_line_plot("data", "ax1")

        
        self.update_plot(None)

        if self.uselims_button.get() and not self.image_plot:
            self.ylim_slider.set([self.ymin, self.ymax])

        if (self.rows == 1 and self.cols == 1):
            self.plot_counter += 1
        else:
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
    
        return colormap.colors
        
        

    def plot_axis_parameters(self, ax, plot_counter=1):
        axis = getattr(self, ax)

        last_row = plot_counter > self.cols*(self.rows-1)
        first_col = (plot_counter-1) % self.cols == 0
        # put the second axis on the right side of the plot
        if ax == "ax2":
            first_col = (plot_counter+self.cols) % self.cols == 0

        # if the hide_ticks option has been chosen in the multiplot, settings window
        if self.ticks_settings == "no ticks":
            axis.set_xticks([])
            axis.set_yticks([])
        elif self.ticks_settings == "smart" and not self.mid_axis_button.get():
            if not last_row: 
                axis.set_xticklabels([])
            if not first_col:
                axis.set_yticklabels([])


        if self.use_grid_lines.get():
            axis.grid(visible=True, which=self.grid_ticks, axis=self.grid_axis)
            axis.minorticks_on()
        else:
            axis.grid(visible=False)

        # place labels only on the left and bottom plots (if there are multiplots)
        if self.uselabels_button.get() == 1:
            if last_row or self.label_settings == "default" or self.mid_axis_button.get(): 
                axis.set_xlabel(self.ent_xlabel.get())
            if first_col or self.label_settings == "default" or self.mid_axis_button.get(): 
                axis.set_ylabel(self.ent_ylabel.get())

        if self.mid_axis_button.get():
            axis.spines[['top', 'right']].set_visible(False)
            axis.spines[['left', 'bottom']].set_position('zero')
            axis.xaxis.set_tick_params(direction='inout')
            axis.yaxis.set_tick_params(direction='inout')
            axis.plot((1), (0), ls="", marker=">", ms=5, color="k",transform=axis.get_yaxis_transform(), clip_on=False)
            axis.plot((0), (1), ls="", marker="^", ms=5, color="k",transform=axis.get_xaxis_transform(), clip_on=False)
        
    def make_line_plot(self, dat, ax):

        if self.multiplot_button.get():
                if self.sub_plot_value == 1:
                    self.two_axis_button.configure(state="disabled")
                else: 
                    self.two_axis_button.configure(state="enabled")
                if self.two_axis_button.get():
                    if self.first_ax2:
                        ax2 = self.ax1.twinx()
                        ax2._get_lines = self.ax1._get_lines  # color cycle of ax1
                        self.ax_container[-1] = (self.ax1, ax2) # Update the container with the twin axis 
                        self.ax2 = ax2
                        self.first_ax2 = False
                    ax = "ax2"

        if self.data_table_button.get():
            self.data = self.read_table_data(self.data_table.get("0.0","end"))

            if self.function_entry.get() != "":
                x = np.linspace(float(self.xval_min_entry.get()), float(self.xval_max_entry.get()),500)
                globals_dict = {"np": np, "x": x}
                y = eval(self.function_entry.get(), globals_dict)
                self.data = np.vstack((x,y)).T

        # Display data in the data table
        self.data_table.delete("0.0", "end")  # delete all text
        for row in self.data:
            self.data_table.insert("end", " \t ".join(map(str, np.round(row,4))) + "\n")

        # Display 1D data
        if len(self.data[0, :]) == 1:
            self.data = np.vstack((range(len(self.data)), self.data[:, 0])).T
        
        data = getattr(self,dat)
        axis = getattr(self,ax)

        axis.tick_params(direction='in')
        
        # create dictionary of the plot key word arguments
        self.plot_kwargs = dict(
            linestyle = self.linestyle,
            marker = self.markers,
            linewidth = self.linewidth
            )
        
        if self.uselabels_button.get(): 
            self.plot_kwargs["label"] = self.ent_legend.get()
            if self.legend_name is not None:
                self.plot_kwargs["label"] += self.optmenu.get()[self.legend_name]

        if self.normalize_button.get(): self.normalize()

        self.ymax = max(np.max(data[:,1]), self.ymax)
        self.ymin = min(np.min(data[:,1]), self.ymin)
        self.xmax = max(np.max(data[:,0]), self.xmax)
        self.xmin = min(np.min(data[:,0]), self.xmin)

        ######### create the plot
        plot = getattr(axis, self.plot_type)
        plot_object, = plot(data[:, 0], moving_average(data[:, 1], self.moving_average), **self.plot_kwargs)
        # axis.fill_between(data[:,0], moving_average(data[:, 1]-data[:,2], self.moving_average), moving_average(data[:, 1]+data[:,2], self.moving_average), alpha=0.1)
        #########
        if self.uselabels_button.get() and (self.ent_legend.get() != "" or self.legend_name is not None): 
            if self.ax2 is not None:
                lines, labels = self.ax1.get_legend_handles_labels()
                lines2, labels2 = self.ax2.get_legend_handles_labels()
                self.ax1.legend(lines+lines2, labels+labels2, **self.legend_type)
            else:
                axis.legend(**self.legend_type)

        # create the fit
        if self.use_fit == 1 and not(self.FFT_button.get() and ax=="ax1") and (not self.image_plot or self.fit_button.get()):
            self.make_fit_plot(ax, dat)

        if self.FFT_button.get() and ax == "ax1":
            self.create_FFT_border_lines(ax)

        self.plot_axis_parameters(ax, plot_counter = self.plot_index)

        return plot_object
            

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

        if self.display_fit_params_in_plot.get():
            axis.text(0.05, 0.9, FitWindow.get_fitted_labels(self.fit_window), ha='left', va='top', transform=axis.transAxes, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.6'), size=8)

    def make_image_plot(self, file_path):
        # used in a onegraph multiplot when we change to an image
        if self.image_plot != self.image_plot_prev and (self.rows == 1 and self.cols == 1) and self.replot:
            return

        self.data = cv2.imdecode(np.fromfile(file_path, np.uint8), cv2.IMREAD_UNCHANGED)
        # if the image has color channels, make sure they have the right order, if all channels are the same, reduce it to one channel
        if len(self.data.shape) == 3:
            self.data = cv2.cvtColor(self.data, cv2.COLOR_BGR2RGB)

            b,g,r = self.data[:,:,0], self.data[:,:,1], self.data[:,:,2]
            if (b==g).all() and (b==r).all():
                self.data = self.data[...,0]
            elif self.convert_rgb_to_gray.get():
                self.data = 0.2989 * r + 0.5870 * g + 0.1140 * b

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
        if self.normalize_button.get(): self.normalize()
        ######### create the plot
        plot = self.ax1.imshow(self.data, **self.plot_kwargs)
        #########
        # use axis labels and add a legend
        if self.uselabels_button.get() and (self.ent_legend.get() != "" or self.legend_name is not None): 
            legend_label = self.ent_legend.get()
            if self.legend_name is not None:
                legend_label += self.optmenu.get()[self.legend_name]
            self.ax1.legend(labels=[legend_label], handles=[self.ax1.plot([],[])[0]], handlelength=0, handleheight=0, handletextpad=0, framealpha=1, **self.legend_type)

        # use a scalebar on the bottom right corner
        if self.use_scalebar.get():
            scalebar = ScaleBar(self.pixel_size*1e-3, box_color="black", pad=0.4, box_alpha=.4, color="white", location="lower right", length_fraction=0.3, sep=2) # 1 pixel = 0.2 meter
            self.fig.gca().add_artist(scalebar)
        
        # use a colorbar
        if self.use_colorbar.get():
            self.cbar = self.fig.colorbar(plot, fraction=0.045, pad=0.01)

        self.plot_axis_parameters("ax1", plot_counter = self.plot_index)
        

    def update_plot(self, val):
        if self.uselims_button.get() == 1: 
            
            if self.two_axis_button.get():
                ax = self.ax2
            else:
                ax = self.ax1 

            self.update_slider_limits()
            self.x_l, self.x_r = self.xlim_slider.get()
            self.y_l, self.y_r = self.ylim_slider.get()

            if isinstance(val, type(None)) and not self.image_plot:
                self.y_r = self.ymax

            if isinstance(val, (tuple, float, str, set, dict, type(None))):
                for (lim_box, limit) in zip(["xlim_lbox", "xlim_rbox", "ylim_lbox", "ylim_rbox"],
                                            ["x_l", "x_r", "y_l", "y_r"]):
                    box = getattr(self,lim_box)
                    box.delete(0,'end')
                    if self.image_plot:
                        box.insert(0,str(int(getattr(self,limit))))
                    else:
                        box.insert(0,str(round(getattr(self,limit),4-len(str(int(getattr(self,limit)))))))

            if self.image_plot: # display images
                ax.set_xlim(left=float(self.x_l), right=float(self.x_r))
                ax.set_ylim(bottom=float(self.y_l), top=float(self.y_r))

            else: # line plot

                ax.set_xlim(left=float(  self.x_l - 0.05 * (self.x_r - self.x_l)), right=float(self.x_r + 0.05 * (self.x_r - self.x_l)))
                ax.set_ylim(bottom=float(self.y_l - 0.05 * (self.y_r - self.y_l)), top=float(  self.y_r + 0.05 * (self.y_r - self.y_l)))
                
                if (self.plot_type == "semilogx" or self.plot_type == "loglog"):
                    xlim_l = calc_log_value(self.x_l - 0.05 * (self.x_r - self.x_l), np.min(abs(self.data[:,0])), np.max(self.data[:,0]))
                    xlim_r = calc_log_value(self.x_r + 0.05 * (self.x_r - self.x_l), np.min(abs(self.data[:,0])), np.max(self.data[:,0]))
                    ax.set_xlim(left=float(xlim_l), right=float(xlim_r))
                if (self.plot_type == "semilogy" or self.plot_type == "loglog"):
                    ylim_l = calc_log_value(self.y_l - 0.05 * (self.y_r - self.y_l), np.min(abs(self.data[:,1])), np.max(self.data[:,1]))
                    ylim_r = calc_log_value(self.y_r + 0.05 * (self.y_r - self.y_l), np.min(abs(self.data[:,1])), np.max(self.data[:,1]))
                    ax.set_ylim(bottom=float(ylim_l), top=float(ylim_r))

        if self.FFT_button.get():
            pass
        if self.fit_button.get():
            try:
                self.fit_xmin_line.set_xdata([self.fit_window.fit_borders_slider.get()[0]])
                self.fit_xmax_line.set_xdata([self.fit_window.fit_borders_slider.get()[1]])
            except:
                if self.lineout_button.get() or self.FFT_button.get():
                    self.create_fit_border_lines("ax_second")
                else:
                    self.create_fit_border_lines("ax1")
            
        self.canvas.draw()
    
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
            if replot and not self.lineout_button.get():
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
            self.folder_path.delete(0, customtkinter.END)
            self.folder_path.insert(0, path)
        global filelist
        filelist = natsorted([fname for fname in os.listdir(self.folder_path.get()) if fname.endswith(file_type_names)])
        self.optmenu.configure(values=filelist)
        self.optmenu.set(filelist[0])
    
    def control_z(self):
        
        if self.replot and (self.rows != 1 or self.cols != 1) and self.plot_counter > 2:
            self.plot_counter -= 1
            self.sub_plot_counter = float('inf')
            ax1, ax2 = self.ax_container.pop()
            ax1.remove()
            if ax2: ax2.remove()

            (self.ax1, self.ax2) = self.ax_container[-1]
        
        elif (self.replot and (self.rows == 1 and self.cols == 1) and self.plot_counter > 2):
            self.ax1.get_lines()[-1].remove()
            # ensure that the color is setback to its previous state by shifting the colormap by the value n determined by the current plot counter and resetting the prop_cycle
            self.plot_counter -= 1
            colors = plt.get_cmap(self.cmap).colors
            n = (self.plot_counter-1) % len(colors)
            self.ax1.set_prop_cycle(cycler(color=colors[n:]+ colors[:n]))

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

    """
    ############################################################################################
    ##############################    Settings Frame     #######################################
    ############################################################################################
    """

    def use_limits(self):
        if not self.initialize_plot_has_been_called: return
        if self.uselims_button.get() == 1:
            row = 4 # 4 from the use_labels
            self.settings_frame.grid()
            self.limits_title = App.create_label(self.settings_frame, text="Limits", font=customtkinter.CTkFont(size=16, weight="bold"), row=row, column=0, columnspan=5, padx=20, pady=(20, 5),sticky=None)
            self.labx = App.create_label(self.settings_frame, text="x limit", column=0, row=row+1)
            self.laby = App.create_label(self.settings_frame, text="y limit", column=0, row=row+2)

            xlim_min, xlim_max, ylim_min, ylim_max = self.reset_limits()

            for i, (lim_lbox, lim_rbox, lim_min, lim_max, lim_slider) in enumerate(zip(["xlim_lbox", "ylim_lbox"], # lim_box - entries
                                                                   ["xlim_rbox", "ylim_rbox"],
                                                                   ["xlim_min", "ylim_min"],
                                                                   ["xlim_max", "ylim_max"],
                                                                   ["xlim_slider", "ylim_slider"])): # lim_slider - slider 
                value_min = locals()[lim_min]
                value_max = locals()[lim_max]

                if not self.image_plot:
                    new_value_min = value_min - (value_max-value_min)*0.2
                    new_value_max = value_max + (value_max-value_min)*0.2
                else: 
                    new_value_min = value_min 
                    new_value_max = value_max


                # create the entry objects "self.xlim_lbox" ...
                setattr(self, lim_lbox, App.create_entry(self.settings_frame, row=row+i+1, column=1, width=50))
                setattr(self, lim_rbox, App.create_entry(self.settings_frame, row=row+i+1, column=4, width=50))
                setattr(self, lim_slider, App.create_range_slider(self.settings_frame, from_=new_value_min, to=new_value_max, command= lambda val=None: self.update_plot(val), row=row+i+1, column =2, width=120, padx=(0,0), number_of_steps=1000, columnspan=2, init_value=[value_min, value_max]))

                # get the name of the created object, first the entry, second the slider
                entryl_widget = getattr(self,lim_lbox)
                entryr_widget = getattr(self,lim_rbox)
                slider_widget = getattr(self,lim_slider)

                #set the properties of the entry object, key release event, set the value to the slider and update the limits,
                # slider_widget = slider_widget is necessary to force the lambda function to capture the current value of slider_widget instead of calling it, when the key is pressed, otherwise slider_widget = ylim_r is used!
                entryl_widget.bind("<KeyRelease>", lambda event, val1=entryl_widget, val2=entryr_widget, slider_widget=slider_widget: (slider_widget.set([float(val1.get()), float(val2.get())]), self.update_plot(val1)))
                entryr_widget.bind("<KeyRelease>", lambda event, val1=entryl_widget, val2=entryr_widget, slider_widget=slider_widget: (slider_widget.set([float(val1.get()), float(val2.get())]), self.update_plot(val2)))
                # insert the limits into the entry with 4 decimals
                entryl_widget.insert(0,str(round(value_min,4-len(str(int(value_min))))))
                entryr_widget.insert(0,str(round(value_max,4-len(str(int(value_max))))))

        else:
            for name in ["xlim_slider","ylim_slider","labx","laby","limits_title", "xlim_lbox", "xlim_rbox", "ylim_lbox", "ylim_rbox"]:
                getattr(self, name).grid_remove()
            self.close_settings_window()
        self.update_plot(None)

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

    def use_labels(self):
        if self.uselabels_button.get() == 1:
            row = 0
            self.settings_frame.grid()
            self.labels_title = App.create_label(self.settings_frame, text="Labels", font=customtkinter.CTkFont(size=16, weight="bold"), row=row, column=0, columnspan=5, padx=20, pady=(20, 5),sticky=None)
            self.ent_ylabel   = App.create_entry(self.settings_frame,column=3, row=row+1, columnspan=2, width=110, placeholder_text="y label")
            self.ent_xlabel, self.ent_label_text  = App.create_entry(self.settings_frame,column=1, row=row+1, columnspan=2, width=110, placeholder_text="x label", text="x / y")
            self.ent_legend, self.ent_legend_text = App.create_entry(self.settings_frame,column=1, row=row+2, columnspan=4, width=110, text="Legend")
            self.legend_settings_button = App.create_button(self.settings_frame, text="Settings", command=lambda: self.open_toplevel(LegendWindow, "Legend Settings"), column=3, row=row+2, columnspan=2, image=self.img_settings, width=110)
        else:
            for name in ["ent_xlabel","ent_label_text","ent_ylabel","ent_legend","ent_legend_text","labels_title", "legend_settings_button"]:
                getattr(self, name).grid_remove()
            self.close_settings_window()
    
    #############################################################
    ##################### multiplot #############################
    #############################################################
    
    def multiplot(self):
        self.replot = not self.replot
        if self.multiplot_button.get() == 1:
            if self.lineout_button.get():
                self.rows = 1
            else:
                self.rows = 2
            self.cols = 2

            row = self.multiplot_row
            self.multiplot_title = App.create_label( self.settings_frame,column=0, row=row, text="Multiplot Grid", font=customtkinter.CTkFont(size=16, weight="bold"), columnspan=5, padx=20, pady=(20, 5),sticky=None)
            self.ent_rows, self.ent_rows_text = App.create_entry( self.settings_frame,column=1, row=row+1, width=50, text="rows")
            self.ent_cols, self.ent_cols_text = App.create_entry( self.settings_frame,column=3, row=row+1, width=50, text="columns")
            self.ent_subplot, self.ent_subplot_text = App.create_entry( self.settings_frame,column=1, row=row+2, width=50, text="subplots")
            self.settings_frame.grid()
            self.two_axis_button.grid()
            self.addplot_button.grid() 
            self.reset_button.grid()
            self.plot_button.grid_remove()

            self.ent_subplot.insert(0,str(1))
            self.ent_rows.insert(0,str(self.rows))
            self.ent_cols.insert(0,str(self.cols))
        else:
            self.plot_button.grid()
            for name in ["ent_rows","ent_rows_text","ent_cols","ent_cols_text","reset_button","addplot_button","multiplot_title","ent_subplot","ent_subplot_text", "two_axis_button"]:
                getattr(self, name).grid_remove()
            self.close_settings_window()
            # reset number of rows and cols for plotting to single plot value
            self.rows = 1
            self.cols = 1
        self.initialize_plot()

    #############################################################
    ######################## Lineout ############################
    #############################################################
        
    def lineout(self):
        if self.image_plot == False:
            self.lineout_button.deselect()

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
            row = 12
            self.lineout_title    = App.create_label( self.settings_frame,column=0, row=row, text="Lineout", font=customtkinter.CTkFont(size=16, weight="bold"), columnspan=5, padx=20, pady=(20, 5),sticky=None)
            self.choose_ax_button = App.create_segmented_button(self.settings_frame, column = 1, row=row+3, values=[" x-line ", " y-line "], command=self.initialize_lineout, columnspan=2, padx=(10,10), sticky="w")
            self.angle_slider     = App.create_slider(self.settings_frame, from_=-90, to=90, command= lambda val=None: self.plot_lineout(val), row=row+1, column =2, width=170, padx=(0,10), columnspan=3, number_of_steps=180)
            self.line_slider      = App.create_slider(self.settings_frame, from_=0, to=len(self.data[0,:]), command= lambda val=None: self.plot_lineout(val), row=row+2, column =2, width=170, padx=(0,10), columnspan=3)
            self.line_entry, self.ent_line_text = App.create_entry(self.settings_frame, row=row+2, column=1, width=50, text="line")
            self.angle_entry, self.ent_angle_text = App.create_entry(self.settings_frame, row=row+1, column=1, width=50, text="angle")
            self.save_lineout_button = App.create_button(self.settings_frame, column = 3, row=row+3, text="Save Lineout", command= lambda: self.save_data_file(self.lineout_data), width=80, columnspan=2)
            self.line_slider.set(self.lineout_yvalue)
            self.line_entry.bind("<KeyRelease>", lambda event, val=self.line_entry, slider_widget=self.line_slider: (slider_widget.set(float(val.get())), self.plot_lineout(val)))
            self.angle_entry.insert(0,str(0))
            self.angle_entry.bind("<KeyRelease>", lambda event, val=self.angle_entry, slider_widget=self.angle_slider: (slider_widget.set(float(val.get())), self.plot_lineout(val)))
            self.choose_ax_button.set(" y-line ")
            self.initialize_lineout(" y-line ")
        else:
            try:
                for name in ["lineout_title", "line_entry", "line_slider", "ent_line_text", "choose_ax_button", "save_lineout_button", "angle_slider", "angle_entry", "ent_angle_text"]:
                    getattr(self, name).grid_remove()
            except:
                return
            self.multiplot_button.configure(state="enabled")
            self.replot = False
            self.close_settings_window()

    def initialize_lineout(self,val):
        self.initialize_plot()
        if self.lineout_button.get() == False: return

        if self.uselims_button.get():
            self.x_lineout_min = int(self.xlim_slider.get()[0])
            self.x_lineout_max = int(self.xlim_slider.get()[1])
            self.y_lineout_min = int(self.ylim_slider.get()[0])
            self.y_lineout_max = int(self.ylim_slider.get()[1])
        else:
            self.x_lineout_min = 0
            self.x_lineout_max = len(self.data[:,0])
            self.y_lineout_min = 0
            self.y_lineout_max = len(self.data[0,:])

        asp_image = len(self.data[self.x_lineout_min:self.x_lineout_max,0]) / len(self.data[0,self.y_lineout_min:self.y_lineout_max])

        if val == " x-line ":
            if self.line_slider.get() >= self.y_lineout_max: self.line_slider.set(self.y_lineout_max-1)
            self.lineout_data = np.array([self.data[int(self.line_slider.get()),self.x_lineout_min:self.x_lineout_max]]).T
            self.line_slider.configure(from_=self.y_lineout_min, to=self.y_lineout_max-1, number_of_steps = len(self.data[0,:])-1)
            
        elif val == " y-line ":
            if self.line_slider.get() >= self.x_lineout_max: self.line_slider.set(self.x_lineout_max-1)
            self.lineout_data = np.array([self.data[self.y_lineout_min:self.y_lineout_max,int(self.line_slider.get())]]).T
            self.line_slider.configure(from_=self.x_lineout_min, to=self.x_lineout_max-1, number_of_steps = len(self.data[:,0])-1)

        (x1, y1), (x2, y2) = self.get_lineout_data()
        self.axline = self.ax1.plot([x1, x2],[y1, y2], 'tab:blue')[0]
        self.axpoint = self.ax1.plot(self.lineout_xvalue, self.lineout_yvalue, 'tab:blue', marker="o")[0]
        
        # reset slider value when the lineout is changed and the value is out of bounds
        self.line_slider.set(self.line_slider.get())
        self.line_entry.delete(0, 'end')
        self.line_entry.insert(0,str(int(self.line_slider.get())))
        
        self.ax_second = self.fig.add_subplot(1,2,2)
        max_value = np.max(self.data)

        self.lineout_plot = self.make_line_plot("lineout_data", "ax_second")
        self.plot_axis_parameters("ax_second")

        # asp = np.diff(self.ax_second.get_xlim())[0] / np.diff(self.ax_second.get_ylim())[0]*asp_image
        asp = np.diff(self.ax_second.get_xlim())[0] / max_value*asp_image

        self.ax_second.set_aspect(abs(asp))
        self.ax_second.set_ylim(0, max_value)

        self.canvas.draw()

    def get_lineout_data(self):
        if self.choose_ax_button.get() == " x-line ":
            self.lineout_xvalue = self.line_slider.get()
        elif self.choose_ax_button.get() == " y-line ":
            self.lineout_yvalue = self.line_slider.get()

        theta = -self.angle_slider.get() 
        (x1, y1), (x2, y2) = get_border_points(self.lineout_xvalue, self.lineout_yvalue, theta, (len(self.data[0,:]), len(self.data[:,0])))

        num = round(np.sqrt((x2-x1)**2 + (y2-y1)**2))
        x, y = np.linspace(x1, x2, num), np.linspace(y1, y2, num)

        # Extract the values along the line, using cubic interpolation
        brightness_value = scipy.ndimage.map_coordinates(self.data, np.vstack((y,x)))

        scale = 1
        if self.convert_pixels.get():
            scale = self.pixel_size

        self.lineout_data = np.vstack([np.arange(0,num*scale,scale), brightness_value]).T
        return (x1, y1), (x2, y2)


    def plot_lineout(self, val):
        self.line_entry.delete(0, 'end')
        self.line_entry.insert(0,str(int(self.line_slider.get())))

        self.angle_entry.delete(0, 'end')
        self.angle_entry.insert(0,str(int(self.angle_slider.get())))
        
        (x1, y1), (x2, y2) = self.get_lineout_data()

        self.axline.set_data([x1, x2],[y1, y2])
        self.axpoint.set_data([self.lineout_xvalue], [self.lineout_yvalue])
        self.lineout_plot.set_data(self.lineout_data[:,0], self.lineout_data[:,1])

        if self.use_fit == 1:
            minimum = np.argmin(abs(self.lineout_data[:,0] - self.fit_window.fit_borders_slider.get()[0]))
            maximum = np.argmin(abs(self.lineout_data[:,0] - self.fit_window.fit_borders_slider.get()[1]))
            self.fit_lineout.set_xdata(self.lineout_data[minimum:maximum,0])
            self.fit_lineout.set_ydata(self.fit_plot(self.function, self.params, self.lineout_data[minimum:maximum,:]))
        self.canvas.draw()

    #############################################################
    ############### Fouriertransformation #######################
    #############################################################
        
    def fourier_trafo(self):
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
            

            self.cols = 2
            self.rows = 1
            self.settings_frame.grid()
            row = 15
            self.FFT_title    = App.create_label( self.settings_frame,column=0, row=row, text="Fourier Transform", font=customtkinter.CTkFont(size=16, weight="bold"), columnspan=5, padx=20, pady=(20, 5),sticky=None)
            self.save_FFT_button = App.create_button(self.settings_frame, column = 3, row=row+1, text="Save FFT", command= lambda: self.save_data_file(self.FFT_data), width=80, columnspan=2)
            self.padd_zeros_textval  = App.create_label( self.settings_frame,column=1, columnspan=2, row=row+1, text=str(0), sticky="n", anchor="n", fg_color="transparent")
            self.settings_frame.rowconfigure(row+1, minsize=50)
            self.padd_zeros_slider = App.create_slider(self.settings_frame, from_=0, to=10, command= self.update_FFT, row=row+1, column=1, columnspan=2, init_value=0, number_of_steps=10, width=120, text="padd 0's", padx=(10,0))
            self.FFT_borders_slider = App.create_range_slider(self.settings_frame, from_=np.min(self.data[:,0]), to=np.max(self.data[:,0]), command= self.update_FFT, row=row+2, column =1, width=120, columnspan=2, init_value=[np.min(self.data[:,0]), np.max(self.data[:,0])], text="FFT lims", padx=(10,0))
            self.initialize_FFT()
        else:
            try:
                for name in ["FFT_title", "save_FFT_button", "FFT_borders_slider", "padd_zeros_textval", "padd_zeros_slider"]:
                    getattr(self, name).grid_remove()
            except:
                return
            self.multiplot_button.configure(state="enabled")
            self.lineout_button.configure(state="enabled")
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
        # self.FFT_plot, = self.ax_second.plot(self.FFT_data[:,0],self.FFT_data[:,1], **self.plot_kwargs)

        # Determine the width of the displayed FT window
        argmax = np.argmax(self.FFT_data[:,1])
        width = np.max(np.argwhere(self.FFT_data[:,1] > 1e-3*np.max(self.FFT_data[:,1]))) - argmax + int(0.01*len(self.FFT_data[:,1])/(self.padd_zeros_slider.get()+1))
        # width = abs(np.argmax(self.FFT_data[:,1]) - np.argmin(abs(self.FFT_data[:,1]-0.5*np.max(self.FFT_data[:,1]))))
        # self.ax_second.set_xlim(self.FFT_data[argmax-10*half_width,0], self.FFT_data[argmax+10*half_width,0])
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

    def normalize(self):
        if self.image_plot:
            self.data = self.data * 0.5/np.mean(self.data)*self.enhance_value
            self.data[self.data>1] = 1
        elif self.change_norm_to_area.get() == False:
            self.data[:, 1:] /= np.max(self.data[:, 1])
        elif self.change_norm_to_area.get() == True:
            self.data[:, 1:] /= (np.sum(self.data[:, 1])*abs(self.data[0,0]-self.data[1,0]))

        self.ymax = max([self.ymax, np.max(self.data[:, 1])])

    def create_toolbar(self):
        toolbar_frame = customtkinter.CTkFrame(master=self.tabview.tab("Show Plots"))
        toolbar_frame.pack()
        # toolbar_frame.grid(column=2, row=3, columnspan=2)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.config(background=self.color)
        toolbar._message_label.config(background=self.color, foreground="white")
        toolbar.winfo_children()[-2].config(background=self.color)
        toolbar.update()
        return toolbar_frame
          
    def skip_rows(self, file_path):
        skiprows = 0
        for line in open(file_path, 'r'):
            if line[0].isdigit():
                break
            else:
                skiprows += 1
        if skiprows > 0: print(f">>>>>>>>>>>>\n There were {skiprows} head lines detected and skipped. \n<<<<<<<<<<<<")
        return skiprows

    def save_figure(self):
        file_name = customtkinter.filedialog.asksaveasfilename()
        if (self.image_plot and self.save_plain_image.get() and file_name.endswith((".pdf",".png",".jpg",".jpeg",".PNG",".JPG",".svg"))):
            if self.uselims_button.get(): 
                plt.imsave(file_name, self.data[int(self.y_l):int(self.y_r),int(self.x_l):int(self.x_r)], cmap=self.cmap_imshow)
            else: 
                plt.imsave(file_name, self.data, cmap=self.cmap_imshow)
        elif file_name.endswith((".pdf",".png",".jpg",".jpeg",".PNG",".JPG",".svg")): 
            self.fig.savefig(file_name, bbox_inches='tight')
        elif file_name.endswith((".dat",".txt",".csv")):
            if self.image_plot:
                np.savetxt(file_name, self.data.astype(int), fmt='%i')
            else:
                np.savetxt(file_name, self.data)

    def open_file(self, file_path):
        try:
            with open(file_path, 'r') as file:
                content = file.read()
        except Exception as e:
            print(f"Error reading the file: {e}")
            return

        if content.count('.') >= content.count(','):
            return '.'
        elif content.count('.') < content.count(','):
            return ','
        else:
            print("No decimal separators (commas or dots) were found in the file.")
    
    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)
        if new_appearance_mode == "Light":
            self.color = "#f2f2f2"
            self.text_color = "black"
        else:
            self.color = "#1a1a1a"
            self.text_color = "white"
        if self.initialize_plot_has_been_called: 
            self.toolbar.pack_forget()
            self.toolbar = self.create_toolbar()

    def toggle_boolean(self, boolean):
        boolean.set(not boolean.get())

    def close_settings_window(self):
        if not (self.uselabels_button.get() or self.uselims_button.get() or self.fit_button.get() or self.multiplot_button.get() or self.lineout_button.get()):
            self.settings_frame.grid_remove()
            
    def open_toplevel(self,cls,var):

        if self.toplevel_window[var] is None or not self.toplevel_window[var].winfo_exists():
            self.toplevel_window[var] = cls(self)  # create window if its None or destroyed
            self.toplevel_window[var].focus()  # focus it
        else:
            self.toplevel_window[var].focus()  # if window exists focus it
            
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
            
    def on_closing(self):
        for name, description in self.tooltips.items():
            tooltip = getattr(self, name+"_ttp")
            tooltip.cleanup()

        quit() # Python 3.12 works
        # self.destroy() # Python 3.9 on Laptop
        # sysexit()

    def on_click(self, event):
        for i, (ax1,ax2) in enumerate(self.ax_container):
            if ax1.in_axes(event):
                # highlighting the subplot
                ax1.set_facecolor('lightgrey')
                self.ax1 = ax1 
                self.ax2 = ax2
                self.plot_counter = i+1
                self.mouse_clicked_on_canvas = True
            else:
                ax1.set_facecolor('white')
        
        # if no axes are clicked
        if event.inaxes == None:
            self.plot_counter = len(self.ax_container) + 1
            self.mouse_clicked_on_canvas = False
        self.canvas.draw_idle()
        
        
"""
############################################################################################
##############################    Settings Window    #######################################
############################################################################################

"""

class SettingsWindow(customtkinter.CTkToplevel): 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry("600x650")
        self.title("Plot Settings")
        self.app = app

        # general values
        self.canvas_ratio = {'Auto': None, '4:3 ratio': 4/3, '16:9 ratio': 16/9, '3:2 ratio': 3/2, '3:1 ratio': 3,'2:1 ratio': 2, '1:1 ratio': 1, '1:2 ratio': 0.5}
        self.ticks_settings = ["smart", "default", "no ticks"]
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
        self.plot_type = {'Linear': 'plot', 'Semi Logarithmic x': 'semilogx', 'Semi Logarithmic y': 'semilogy', 'Log-Log plot': 'loglog'}

        App.create_label(self, column=1, row=0, columnspan=2, text="General settings", font=customtkinter.CTkFont(size=16, weight="bold"),padx=20, pady=(20, 5), sticky=None)
        App.create_label(self, column=1, row=4, columnspan=2, text="Image plot settings", font=customtkinter.CTkFont(size=16, weight="bold"),padx=20, pady=(20, 5), sticky=None)
        App.create_label(self, column=1, row=10, columnspan=2, text="Line plot settings", font=customtkinter.CTkFont(size=16, weight="bold"),padx=20, pady=(20, 5), sticky=None)

        # general values 
        self.canvas_ratio_list   = App.create_optionMenu(self, column=1, row=1, width=110, columnspan=1, values=list(self.canvas_ratio.keys()), text="Canvas Size", command=self.set_plot_settings)
        self.canvas_width        = App.create_entry(self,   column=2, row=1, width=70, columnspan=1, placeholder_text="10 [cm]", sticky='w')
        self.ticks_settings_list = App.create_optionMenu(self, column=1, row=2, width=110, columnspan=1, values=self.ticks_settings, text="Ticks, Label settings", command=self.set_plot_settings)
        self.label_settings_list = App.create_optionMenu(self, column=2, row=2, width=70, columnspan=1, values=self.label_settings, command=self.set_plot_settings)

        # values for image plots, we need the dummy labels to access self.pixel_size at the tooltip
        self.pixel_size, self.dummy_label = App.create_entry(self,   column=1, row=5, columnspan=2, text="Pixel size [µm]", placeholder_text="7.4 µm", sticky='w')
        self.label_dist, self.dummy_label = App.create_entry(self,   column=1, row=6, columnspan=2, text="Label step [mm]", placeholder_text="1 mm", sticky='w')
        self.aspect             = App.create_optionMenu(self, column=1, row=7, columnspan=2, values=self.aspect_values, text="Aspect",  command=self.set_plot_settings)
        self.cmap_imshow_list   = App.create_optionMenu(self, column=1, row=8, columnspan=2, values=self.cmap_imshow,                text="Colormap",command=self.set_plot_settings)
        self.enhance_slider     = App.create_slider(  self,   column=1, row=9, columnspan=2, init_value=1, from_=0.1, to=2, 
                                                    command= lambda value, var="enhance_var": self.update_slider_value(value, var), text="Enhance value", number_of_steps=100)

        # values for line plots
        self.plot_type_list     = App.create_optionMenu(self, column=1, row=11, columnspan=2, values=list(self.plot_type.keys()),text="Plot type",command=self.set_plot_settings)
        self.linestyle_list     = App.create_optionMenu(self, column=1, row=12, width=110, columnspan=1, values=list(self.linestyle.keys()), text="Line style & Marker", command=self.set_plot_settings)
        self.marker_list        = App.create_optionMenu(self, column=2, row=12, width=70, columnspan=1, values=list(self.markers.keys()),     command=self.set_plot_settings)
        self.cmap_list          = App.create_optionMenu(self, column=1, row=13, width=110, columnspan=1, values=self.cmap,                text="Colormap",   command=self.set_plot_settings)
        self.single_colors_list = App.create_optionMenu(self, column=2, row=13, width=70, columnspan=1, values=list(self.single_colors.keys()), command=self.set_plot_settings)
        self.grid_lines_list    = App.create_optionMenu(self, column=1, row=14, width=110, values=self.grid_lines, text="Grid",  command=self.set_plot_settings)
        self.grid_axis_list     = App.create_optionMenu(self, column=2, row=14, width=70, values=self.grid_axis, command=self.set_plot_settings)
        self.moving_av_list     = App.create_optionMenu(self, column=1, row=15, columnspan=2, values=self.moving_average,      text="Average",   command=self.set_plot_settings)
        self.cmap_length_list   = App.create_optionMenu(self, column=2, row=13, width=70, columnspan=1, values=self.cmap_length, command=self.set_plot_settings)
        self.linewidth_slider   = App.create_slider(  self,  column=1, row=16, columnspan=2, init_value=1, from_=0.1, to=2, 
                                                    command= lambda value, var="lw_var": self.update_slider_value(value, var), text="line width", number_of_steps=100)

        self.reset_button           = App.create_button(self, column=4, row=0, text="Reset Settings",   command=self.reset_values, width=130, pady=(20,5))
        self.convert_pixels_button  = App.create_switch(self, column=4, row=5, text="Convert Pixels",   command=lambda: self.toggle_boolean(self.app.convert_pixels))
        self.scale_switch_button    = App.create_switch(self, column=4, row=6, text="Use Scalebar",     command=lambda: self.toggle_boolean(self.app.use_scalebar))
        self.cbar_switch_button     = App.create_switch(self, column=4, row=7, text="Use Colorbar",     command=lambda: self.toggle_boolean(self.app.use_colorbar))
        self.convert_rgb_button     = App.create_switch(self, column=4, row=8, text="Convert RGB to gray",command=lambda: self.toggle_boolean(self.app.convert_rgb_to_gray))
        self.save_plain_img_button  = App.create_switch(self, column=4, row=9, text="Save plain images",command=lambda: self.toggle_boolean(self.app.save_plain_image))
        self.hide_params_button     = App.create_switch(self, column=4, row=11, text="Hide fit params",  command=lambda: self.toggle_boolean(self.app.display_fit_params_in_plot))
        self.norm_switch_button     = App.create_switch(self, column=4, row=12, text="Normalize 'Area'", command=lambda: self.toggle_boolean(self.app.change_norm_to_area))
        self.grid_lines_button      = App.create_switch(self, column=4, row=13, text="Use Grid",         command=lambda: self.toggle_boolean(self.app.use_grid_lines))
        self.hide_ticks_button      = App.create_switch(self, column=4, row=14, text="hide ticks",       command=lambda: self.toggle_boolean(self.app.hide_ticks))
        
        self.cmap_length_list.configure(state="disabled")
        self.single_colors_list.grid_remove()
        #tooltips
        self.tooltips = {"canvas_ratio_list":"Determine the size of the canvas. This is useful for saving the plots in a reproducible manner\n- Auto: Resize the canvas to fill the full screen (standard)\n- 4:3 ratio: Set the size of the canvas in 4:3 format, the width [cm] can be set with the entry box on the right",
                    "canvas_width":     "Set the width of the canvas in cm.\n- Works only, if Canvas Size is not set to 'Auto'",
                    "pixel_size":       "Set the size of a single pixel\n- the tick labels can be converted to real lengths by checking 'Convert Pixels'.",
                    "label_dist":       "Set, at which points a new label is placed", 
                    "aspect":           "Set the aspect of the image\n- 'equal' - original image aspect, no pixel distortion\n- 'auto' - fill the image to the size of the window, produces image distortion.",
                    "cmap_imshow_list": "Choose between different colormaps", 
                    "enhance_slider":   "Activation by checking 'Normalize'\n- Enhance the contrast of the images by dividing all pixel by the mean pixel value and multiplying them with the 'Enhance value'. Values > 255 will be set to 255 (white)",
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
                    "norm_switch_button": "Only for line plots:\n- Off: Normalize to a maximum value of 1\n- On: Normalize to area of 1", 
                    "grid_lines_button": "Use Gridlines", 
                    "hide_ticks_button": "hide all ticks and tick-labels"
                    }
        
        for name, description in self.tooltips.items():
            setattr(self, name+"_ttp", CreateToolTip(getattr(self, name), description))

        # Slider labels    
        self.enhance_var = customtkinter.StringVar()  # StringVar to hold the label value
        self.lw_var = customtkinter.StringVar()  # StringVar to hold the label value
        App.create_label(self, textvariable=self.enhance_var, column=3, row=9, width=30, anchor='w', sticky='w')
        App.create_label(self, textvariable=self.lw_var, column=3, row=16, width=30, anchor='w', sticky='w')
        
        self.canvas_width.bind("<KeyRelease>", self.set_plot_settings)
        self.pixel_size.bind("<KeyRelease>", self.set_plot_settings)
        self.label_dist.bind("<KeyRelease>", self.set_plot_settings)

        #set initial values
        self.init_values()
        
    # initiate current values
    def init_values(self):
        self.canvas_ratio_list.set(list(self.canvas_ratio.keys())[list(self.canvas_ratio.values()).index(self.app.canvas_ratio)])
        self.canvas_width.insert(0,str(self.app.canvas_width))
        self.ticks_settings_list.set(self.app.ticks_settings)
        self.label_settings_list.set(self.app.label_settings)
        self.pixel_size.insert(0,str(self.app.pixel_size*1e3))
        self.label_dist.insert(0,str(self.app.label_dist))
        self.aspect.set(self.app.aspect)
        self.cmap_imshow_list.set(self.app.cmap_imshow)
        self.linestyle_list.set(list(self.linestyle.keys())[list(self.linestyle.values()).index(self.app.linestyle)])
        self.marker_list.set(list(self.markers.keys())[list(self.markers.values()).index(self.app.markers)])
        self.plot_type_list.set(list(self.plot_type.keys())[list(self.plot_type.values()).index(self.app.plot_type)])
        self.single_colors_list.set(list(self.single_colors.keys())[list(self.single_colors.values()).index(self.app.single_color)])
        self.cmap_list.set(self.app.cmap)
        self.grid_lines_list.set(self.app.grid_ticks)
        self.grid_axis_list.set(self.app.grid_axis)
        self.moving_av_list.set(self.app.moving_average)
        self.cmap_length_list.set(self.app.cmap_length)
        self.enhance_slider.set(self.app.enhance_value)
        self.linewidth_slider.set(self.app.linewidth)
        self.update_slider_value(self.app.enhance_value, "enhance_var")
        self.update_slider_value(self.app.linewidth, "lw_var")

        if self.app.use_scalebar.get() == True: self.scale_switch_button.select()
        if self.app.use_colorbar.get() == True: self.cbar_switch_button.select()
        if self.app.convert_pixels.get() == True: self.convert_pixels_button.select()
        if self.app.display_fit_params_in_plot.get() == False: self.hide_params_button.select()
        if self.app.change_norm_to_area.get() == True: self.norm_switch_button.select()
        if self.app.convert_rgb_to_gray.get(): self.convert_rgb_button.select()
        if self.app.save_plain_image.get(): self.save_plain_img_button.select()
        if self.app.use_grid_lines.get(): self.grid_lines_button.select()
        if self.app.hide_ticks.get(): self.hide_ticks_button.select()

    # reset to defaul values
    def reset_values(self):
        # Here we define the standard values!!!
        self.app.canvas_width = 17
        self.app.pixel_size = 7.4e-3
        self.app.label_dist = 1
        self.ticks_settings_list.set(self.ticks_settings[0])
        self.label_settings_list.set(self.label_settings[0])
        self.canvas_width.delete(0, 'end')
        self.canvas_width.insert(0,str(self.app.canvas_width))
        self.pixel_size.delete(0, 'end')
        self.pixel_size.insert(0,str(self.app.pixel_size*1e3))
        self.label_dist.delete(0, 'end')
        self.label_dist.insert(0,str(self.app.label_dist))
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

        self.enhance_slider.set(1)
        self.linewidth_slider.set(1) # before next line!
        self.update_slider_value(1, "enhance_var")
        self.update_slider_value(1, "lw_var")
        
        
    def update_slider_value(self, value, var_name):
        variable = getattr(self, var_name)
        variable.set(str(round(value,2)))
        self.set_plot_settings(None)
        
    def set_plot_settings(self, val):
        try:
            self.app.canvas_width=float(self.canvas_width.get())
            self.app.pixel_size=float(self.pixel_size.get())*1e-3
            self.app.label_dist=float(self.label_dist.get())
        except: pass 
        self.app.aspect=self.aspect.get()
        self.app.cmap_imshow=self.cmap_imshow_list.get()
        self.app.enhance_value=self.enhance_slider.get()

        self.app.canvas_ratio=self.canvas_ratio[self.canvas_ratio_list.get()]
        self.app.label_settings = self.label_settings_list.get()
        self.app.ticks_settings = self.ticks_settings_list.get()
        self.app.linestyle=self.linestyle[self.linestyle_list.get()]
        self.app.markers=self.markers[self.marker_list.get()]
        self.app.cmap=self.cmap_list.get()
        self.app.linewidth=self.linewidth_slider.get()
        self.app.moving_average = int(self.moving_av_list.get())
        self.app.grid_ticks = self.grid_lines_list.get()
        self.app.grid_axis = self.grid_axis_list.get()
        self.app.plot_type=self.plot_type[self.plot_type_list.get()]
        self.app.cmap_length = int(self.cmap_length_list.get())
        self.app.single_color = self.single_colors[self.single_colors_list.get()]

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

    def toggle_boolean(self, boolean):
        boolean.set(not boolean.get())


"""
############################################################################################
##############################     Legend Window     #######################################
############################################################################################

"""

class LegendWindow(customtkinter.CTkToplevel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry("500x250")
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
        
        self.legend_name = {'Empty': None, 
                            'Full name': slice(0,None), 
                            'Remove file type': slice(0,-4),
                            'Custom name': slice(None,None),
                            }

        App.create_label(self, column=1, row=0, columnspan=2, text="Legend Settings", font=customtkinter.CTkFont(size=16, weight="bold"),padx=20, pady=(20, 5), sticky=None)

        # values for line plots
        self.reset_button       = App.create_button(self, column=1, row=5, text="Reset Settings",   command=self.reset_values, width=150)
        self.legend_type_list   = App.create_optionMenu(self, column=1, row=1, columnspan=2, values=list(self.legend_type.keys()), text="Legend location", command=self.set_plot_settings)
        self.legend_name_list   = App.create_optionMenu(self, column=1, row=2, columnspan=2, values=list(self.legend_name.keys()), text="Legend name", command=self.set_plot_settings)
        self.name_slice_slider  = App.create_range_slider(self, from_=0, to=len(self.app.optmenu.get()), text="slice file name", command= lambda value, var="slice_var": self.update_slider_value(value, var), row=3, column =1, width=200, columnspan=2, init_value=[0, len(self.app.optmenu.get())])

        self.slice_var = customtkinter.StringVar()  # StringVar to hold the label value
        App.create_label(self, textvariable=self.slice_var, column=1, row=4, width=50, sticky=None)
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

    # reset to defaul values
    def reset_values(self):
        self.legend_type_list.set(next(iter(self.legend_type)))
        self.legend_name_list.set(next(iter(self.legend_name)))
        # Here we define the standard values!!!
        
        
    def update_slider_value(self, value, var_name):
        variable = getattr(self, var_name)
        (x1, x2) = value 
        slice_object = slice(int(x1), int(x2))
        variable.set(self.app.optmenu.get()[slice_object])
        self.legend_name['Custom name'] = slice_object
        self.set_plot_settings(None)
        
    def set_plot_settings(self, val):
        # self.app.aspect=self.aspect.get()
        self.app.legend_type = self.legend_type[self.legend_type_list.get()]
        self.app.legend_name = self.legend_name[self.legend_name_list.get()]

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
        self.labels_title = App.create_label(app.settings_frame, text="Fit Function", font=customtkinter.CTkFont(size=16, weight="bold"),row=self.row , column=0, columnspan=5, padx=20, pady=(20, 5),sticky=None)
        self.function = {'Gaussian': gauss, 'Gaussian 3rd': gauss3, 'Lorentz': lorentz, 'Linear': linear, 'Quadratic': quadratic, 'Exponential': exponential, 'Square Root': sqrt}

         #Step 3: Convert Matplotlib image to Pillow image
        latex_code = r'E=mc^2'
        image = self.render_latex_to_image(latex_code)
        # photo = ImageTk.PhotoImage(image)
        self.image_formula = customtkinter.CTkImage(dark_image=image, size=(image.width, image.height))

        self.function_label = App.create_label(app.settings_frame, text="", column=1, row=self.row +1, width=80, columnspan=4, anchor='e', sticky="w")
        self.function_label_list = {'Gaussian': ["f(x) = a·exp(-(x-b)²/(2·c²)) + d", r"f(x) = a \cdot \exp\left(-\,\dfrac{(x-b)^2}{(2·c^2)}\right) + d"], 
                                    'Gaussian 3rd': ["f(x) = a·exp(-|x-b|³/(2·c²)) + d", r"f(x) = a \cdot \exp\left(-\,\dfrac{|x-b|^3}{(2·c^2)}\right) + d"], 
                                    'Lorentz': ["f(x) = a/[(x²-b²)² + c²·b²]", r"f(x) = \dfrac{a}{(x^2 - b^2)^2 + c^2 \cdot b^2}"],
                                    'Linear': ["f(x) = a·x + b", r"f(x) = a\cdot x + b"], 
                                    'Quadratic': ["f(x) = a·x² + b·x + c", r"f(x) = a\cdot x^2 + b\cdot x+c"], 
                                    'Exponential': ["f(x) = a·exp(b·x) + c", r"f(x) = a\cdot \exp(b\cdot x) +c"], 
                                    'Square Root': ["f(x) = a·sqrt(b·x) + c", r"f(x) = a\sqrt{b\cdot x} + c"]}
        self.function_list_label = App.create_label(app.settings_frame,text="Fit", column=0,row=self.row +2, sticky="e")
        self.function_list = App.create_combobox(app.settings_frame, values=list(self.function.keys()), command=self.create_params, width=110, column=1, row=self.row +2, columnspan=2, sticky="w")
        self.function_list.set('Gaussian')

        if app.lineout_button.get():
            data = app.lineout_data
        else:
            data = app.data

        self.save_fit_button = App.create_button(app.settings_frame, column = 3, row=self.row+2, text="Save fit", command= lambda: app.save_data_file(np.vstack([data[:,0], app.fit_plot(app.function, app.params, data)]).T), width=80, columnspan=2)
        if (not app.image_plot or app.lineout_button.get()):
            self.fit_borders_slider = App.create_range_slider(app.settings_frame, from_=np.min(data[:,0]), to=np.max(data[:,0]), command= lambda val=None: app.update_plot(val), row=self.row+3, column =1, width=180, padx=(10,10), columnspan=4, init_value=[np.min(data[:,0]), np.max(data[:,0])])

        self.params = []
        self.error = []
        self.create_params('Gaussian')
        # self.set_fit_params()
        app.settings_frame.grid()
    
    #########################################################
    # Step 2: Function to render LaTeX using Matplotlib
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
    ##################################################################

    def create_params(self, function_name):
        # set the label of the function, e.g f(x) = a*x+b
        self.function_name = function_name
        # self.function_label.configure(text = self.function_label_list[self.function_list.get()])
        latex_code = self.function_label_list[self.function_list.get()][1]
        image = self.render_latex_to_image(latex_code)
        self.function_label.configure(image=customtkinter.CTkImage(dark_image=image, size=(image.width, image.height)))
        
        #Determine the number of arguments of the used function
        self.number_of_args = len(signature(self.function[function_name]).parameters) - 1
        
        for i,name in enumerate(["a","b","c","d","e","f"]):
            self.create_parameter(name, i)

        if self.function_name == "Gaussian":
            self.widget_dict['str_var_FWHM'] = customtkinter.StringVar()
        self.set_fit_params(app.data)
        self.row += 2 + self.number_of_args
    
    def create_parameter(self, name, arg_number):
        # try deleting the parameters of previous fit function
        try: 
            for attribute in ["fit", "fitlabel", "fitted"]:
                self.widget_dict[f'{attribute}_{name}'].grid_remove()
        except: pass
        
        if arg_number < self.number_of_args:
            self.widget_dict[f'fit_{name}'], self.widget_dict[f'fitlabel_{name}'] = App.create_entry(app.settings_frame, column=1, row=self.row + arg_number + 4, width=80, placeholder_text="0", sticky='w', columnspan=2, text=name)
            self.widget_dict[f'str_var_{name}'] = customtkinter.StringVar() # StringVar to hold the label value
            self.widget_dict[f'fitted_{name}'] = App.create_label(app.settings_frame, textvariable=self.widget_dict[f'str_var_{name}'], column=3, width=50, row=self.row + arg_number + 4, padx=(0, 20), columnspan=2)
    
    def set_fitted_values(self, params, error):
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
    
    # display values in the textbox, function is used in app.plot()
    def get_fitted_labels(self):
        string = self.function_label_list[self.function_list.get()][0] 
        
        for i,name in enumerate(['a','b','c','d', 'FWHM']):
            if i >= self.number_of_args: break
            round_digit = -int(np.floor(np.log10(abs(np.sqrt(self.error[i,i])))))
            string += "\n" + name +" = " + str(round(self.params[i],round_digit))+" ± "+str(round(np.sqrt(self.error[i,i]),round_digit))
        return string
        
    def set_fit_params(self, data):
        self.app.params = []
        for i,name in enumerate(['a','b','c','d']):
            if i >= self.number_of_args: break   # take only the right number of arguments
            try:
                value = float(self.widget_dict[f'fit_{name}'].get())
                self.app.params.append(value)
            except ValueError:
                self.app.params.append(1)
                if (self.function_name in ["Gaussian","Gaussian 3rd","Lorentz"]) and i == 1:
                    self.app.params[1] = data[np.argmax(data[:,1]),0]
                    self.app.params[0] = np.max(data[:,1])

        self.app.function = self.function[self.function_list.get()]
        # app.initialize_plot()
        self.app.use_fit = app.fit_button.get()
        
    def close_window(self):
        for i,name in enumerate(["a","b","c","d"]):
            try: 
                for attribute in ["fit", "fitlabel", "fitted"]:
                    self.widget_dict[f'{attribute}_{name}'].grid_remove()
                self.widget_dict['fitted_FWHM'].grid_remove(), self.widget_dict['fittedlabel_FWHM'].grid_remove()
            except: pass
    
        self.labels_title.grid_remove()
        self.function_list.grid_remove()
        self.function_list_label.grid_remove()
        self.function_label.grid_remove()
        self.fit_borders_slider.grid_remove()
        self.app.use_fit = app.fit_button.get()

'''
############################################################################################
##############################     Table  Window     #######################################
############################################################################################
'''
class Table(customtkinter.CTkScrollableFrame):
    def __init__(self, master, data, **kwargs):
        super().__init__(master, **kwargs)

        self.columns = ("x", "y")
        self.data = data

        self.text_widget = customtkinter.CTkTextbox(self, padx=10, pady=5)
        self.text_widget.grid(row=0, column=0, sticky="nsew")
        self.grid_columnconfigure(0, weight=1)
        
        self.text_widget.insert("1.0", "\t".join(self.columns) + "\n")

        for row in self.data:
            # self.insert_data(row)
            self.text_widget.insert("end", "\t".join(map(str, row)) + "\n")

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
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 0
        y += self.widget.winfo_rooty() + 40
        # creates a toplevel window
        self.tw = customtkinter.CTkToplevel(self.widget, fg_color = ["#f9f9fa","#343638"]) #["#979da2","#565b5e"]
        # Leaves only the label and removes the app window
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = customtkinter.CTkLabel(self.tw, text=self.text + "\nPress F1 to deactivate tool tips", justify='left', wraplength = self.wraplength, fg_color = "transparent", padx=10, pady=5)
        label.pack()

    def hidetip(self):
        tw = self.tw
        self.tw= None
        if tw:
            tw.destroy()
    
    def cleanup(self):
        self.unschedule()
        self.hidetip()


if __name__ == "__main__":
    app = App()
    app.state('zoomed')
    app.protocol("WM_DELETE_WINDOW", app.on_closing)

    app.bind("<Control-z>", lambda x: app.control_z())
    app.bind("<F1>", toggle_bool)
    app.mainloop()