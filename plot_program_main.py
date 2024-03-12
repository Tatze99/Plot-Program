# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 14:28:26 2022

@author: Martin

to do: implement minor ticks as option
"""
# import tkinter
import os
from CTkRangeSlider import *
import customtkinter
from CTkTable import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib_scalebar.scalebar import ScaleBar
# from matplotlib.figure import Figure
import ctypes
# import io
import pandas as pd
import matplotlib
from cycler import cycler
from scipy.optimize import curve_fit as cf
from scipy.interpolate import interp1d
from inspect import signature # get number of arguments of a function
import cv2 # image fourier transform
from PIL import Image, ImageTk

version_number = "24/02"
plt.style.use('default')
matplotlib.rc('font', family='serif')
matplotlib.rc('font', serif='Times New Roman')

ctypes.windll.shcore.SetProcessDpiAwareness(1)

Standard_path = os.path.dirname(os.path.abspath(__file__))
filelist = [fname for fname in os.listdir(Standard_path) if fname.endswith(('.csv', '.dat', '.txt', '.png', '.jpg'))]

def exponential(x, a, b, c):
    return a * np.exp(b * x) + c

def lorentz(x, a, w0, gamma):
    return a / ((x**2 - w0**2)**2 + gamma**2*w0**2)

def gauss(x, a, b, c, d):
    return a * np.exp(-(x - b)**2 / (2 * c**2)) + d

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

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        customtkinter.set_appearance_mode("dark")
        customtkinter.set_default_color_theme("dark-blue")
        self.title("Graph interactive editor for smooth visualization GIES v."+version_number)
        self.geometry("1280x720")
        self.replot = False
        self.initialize_ui_images()
        self.initialize_ui()
        self.initialize_plot_has_been_called = False
        self.folder_path.insert(0, Standard_path)
        self.toplevel_window = None
        # hide the multiplot button
        self.addplot_button.grid_remove() 
        self.reset_button.grid_remove()
        self.initialize_variables()
        
    def initialize_ui_images(self):
        # self.img_settings = ImageTk.PhotoImage(
        #         Image.open(os.path.join(Standard_path,"ui_images","options.png")).resize((12, 12)),
        #         Image.Resampling.LANCZOS,
        #     )
        
        self.img_settings = customtkinter.CTkImage(dark_image=Image.open(os.path.join(Standard_path,"ui_images","options.png")), size=(15, 15))
        self.img_save = customtkinter.CTkImage(dark_image=Image.open(os.path.join(Standard_path,"ui_images","save_white.png")), size=(15, 15))
        self.img_folder = customtkinter.CTkImage(dark_image=Image.open(os.path.join(Standard_path,"ui_images","folder.png")), size=(15, 15))
        
    def initialize_variables(self):
        self.ymax = 0
        self.ax1 = None
        self.plot_counter = 1
        self.color = "#1a1a1a" # toolbar
        self.reset_plots = True

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
        self.one_multiplot = customtkinter.BooleanVar(value=True)
        self.rows = 1
        self.cols = 1
        
        
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
        
        self.plot_button    = App.create_button(frame, text="Reset graph",             command=self.initialize, column=0, row=5, pady=(15,5))
        self.reset_button   = App.create_button(frame, text="Reset",                  command=self.initialize_plot, column=0, row=5, width=90, padx = (10,120), pady=(15,5))   # Multiplot button
        self.addplot_button = App.create_button(frame, text="Multiplot",              command=self.plot,            column=0, row=5, width=90, padx = (120,10), pady=(15,5))   # Multiplot button
        self.load_button    = App.create_button(frame, text="Load Folder", command=self.read_file_list,  column=0, row=1, image=self.img_folder)
        self.save_button    = App.create_button(frame, text="Save Figure", command=self.save_figure,     column=0, row=3,  image=self.img_save)
        self.set_button     = App.create_button(frame, text="Plot settings",          command=lambda: self.open_toplevel(SettingsWindow), column=0, row=4, image=self.img_settings)
        self.prev_button    = App.create_button(frame, text="<<",                     command=lambda: self.change_plot(self.replot,-1), column=0, row=6, width=90, padx = (10,120))
        self.next_button    = App.create_button(frame, text=">>",                     command=lambda: self.change_plot(self.replot,1), column=0, row=6, width=90, padx = (120,10))
        
        #switches
        self.multiplot_button = App.create_switch(frame, text="Multiplot",  command=self.multiplot,   column=0, row=7, padx=20, pady=(10,5))
        self.uselabels_button = App.create_switch(frame, text="Use labels", command=self.use_labels,  column=0, row=8, padx=20)
        self.uselims_button   = App.create_switch(frame, text="Use limits", command=self.use_limits,  column=0, row=9, padx=20)
        self.fit_button       = App.create_switch(frame, text="Use fit",    command=lambda: (self.open_widget(FitWindow), self.initialize), column=0, row=10, padx=20)
        self.normalize_button = App.create_switch(frame, text="Normalize",  command=self.normalize_setup,    column=0, row=11, padx=20)
        self.lineout_button   = App.create_switch(frame, text="Lineout",  command=self.lineout,    column=0, row=12, padx=20)
        self.FFT_button       = App.create_switch(frame, text="FFT",  command=self.fourier_trafo,    column=0, row=13, padx=20)
        
        for name in ["multiplot_button", "uselabels_button", "uselims_button", "fit_button", "normalize_button", "lineout_button"]:
                getattr(self, name).configure(state="disabled")

        self.settings_frame = customtkinter.CTkFrame(self, width=1, height=600, corner_radius=0)
        self.settings_frame.grid(row=0, column=4, rowspan=999, sticky="nesw")
        self.columnconfigure(2,weight=1)
        # self.columnconfigure(1,weight=1)
        
        self.appearance_mode_label = App.create_label(frame, text="Appearance Mode:", row=16, column=0, padx=20, pady=(10, 0), sticky="w")
        self.appearance_mode_optionemenu = App.create_optionMenu(frame, values=["Dark","Light", "System"], command=self.change_appearance_mode_event, width=200, row=17, column=0, padx=20, pady=(5,10))
        
        #entries
        self.folder_path = self.create_entry(column=2, row=0, text="Folder path", columnspan=2, width=600, padx=10, pady=10, sticky="w")
        
        #dropdown menu
        self.optmenu = self.create_combobox(values=filelist, text="File name",row=1, column=2, columnspan=2, width=600, sticky="w")
        if not filelist == []: self.optmenu.set(filelist[0])


    def create_label(self, row, column, width=20, text=None, anchor='e', sticky='e', textvariable=None, padx=(5,5), pady=None, font=None, columnspan=1, fg_color=None, **kwargs):
        label = customtkinter.CTkLabel(self, text=text, textvariable=textvariable, width=width, anchor=anchor, font=font, fg_color=fg_color,  **kwargs)
        label.grid(row=row, column=column, sticky=sticky, columnspan=columnspan, padx=padx, pady=pady, **kwargs)
        return label

    def create_entry(self, row, column, width=200, text=None, columnspan=1, padx=10, pady=5, placeholder_text=None, sticky='w', **kwargs):
        entry = customtkinter.CTkEntry(self, width, placeholder_text=placeholder_text, **kwargs)
        entry.grid(row=row, column=column, columnspan=columnspan, padx=padx, pady=pady, sticky=sticky, **kwargs)
        if text is not None:
            App.create_label(self, text=text, column=column-1, row=row, width=80, anchor='e')
        return entry

    def create_button(self, text, command, row, column, image=None, width=200, columnspan=1, padx=10, pady=5, sticky=None, **kwargs):
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
            text_widget.insert("1.0", "\t".join(header) + "\n")

        for row in data:
            text_widget.insert("end", " \t ".join(map(str, np.round(row,2))) + "\n")
        return text_widget
    
    #################################################
    ###############Do the plots######################
    #################################################

    def initialize(self):
        if self.lineout_button.get():
            self.initialize_lineout(self.lineout_xy)
        elif self.FFT_button.get():
            self.initialize_FFT()
        else:
            self.initialize_plot()

    def initialize_plot(self):
        
        # Clear the previous plot content
        if self.ax1 is not None:
            self.ax1.clear()  
            self.canvas.get_tk_widget().destroy()
            self.toolbar.destroy()
            for widget_key, widget in self.lists.items():
                widget.destroy()
            for widget_key, widget in self.listnames.items():
                widget.destroy()
            self.plot_counter = 1
            plt.close(self.fig)
            
        
        self.lists = {}
        self.listnames = {}
        if not self.initialize_plot_has_been_called:
            self.initialize_plot_has_been_called = True
            for name in ["multiplot_button", "uselabels_button", "uselims_button", "fit_button", "normalize_button", "lineout_button"]:
                getattr(self, name).configure(state="enabled")
            
        if self.replot and not self.lineout_button.get() and not self.FFT_button.get():
            self.rows=float(self.ent_rows.get())
            self.cols=float(self.ent_cols.get())

        self.fig = plt.figure(figsize=(6,4.5),constrained_layout=True, dpi=150)    
        # self.ax1.tick_params(direction="in", color="gray")
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tabview.tab("Show Plots"))
        # self.canvas.get_tk_widget().grid(row=2, column=2, columnspan=2)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar = self.create_toolbar()
        self.ymax = 0
        self.plot()
        if ((self.uselims_button.get() and self.reset_plots and not self.lineout_button.get() and not self.FFT_button.get()) or (self.uselims_button.get() and self.image_plot != self.image_plot_prev)): 
            # make sure the sliders are
            self.update_limits()
            xlim_min, xlim_max, ylim_min, ylim_max = self.reset_limits()

            self.xlim_slider.set([xlim_min, xlim_max])
            self.ylim_slider.set([ylim_min, ylim_max])

            self.update_plot("Update Limits")
        
    def plot(self):
        file_path = os.path.join(self.folder_path.get(), self.optmenu.get())

        # Decide if there is an image to process or a data file
        self.image_plot_prev = self.image_plot
        if (".png" in file_path or ".jpg" in file_path):
            self.image_plot = True 
            self.one_multiplot.set(False)
            try: 
                self.switch_multiplt.deselect()
                self.switch_multiplt.configure(state="disabled")
            except: pass
        else:
            self.image_plot = False
            try:
                self.switch_multiplt.configure(state="enabled")
            except: pass

        # multiplots but in several subplots
        if self.replot and not self.one_multiplot.get():
            if self.plot_counter > self.rows*self.cols: return
            self.ax1 = self.fig.add_subplot(int(self.rows),int(self.cols),self.plot_counter)
        # multiplot but same axis, no image plot!
        elif (self.replot and self.one_multiplot.get() and not self.image_plot and self.plot_counter == 1) or not self.replot:
            self.ax1 = self.fig.add_subplot(1,1,1)
            self.ax1.set_prop_cycle(cycler(color=plt.get_cmap(self.cmap).colors))

        # create the plot depending if we have a line or image plot
        if self.image_plot: self.make_image_plot(file_path)
        else:
            if self.lineout_button.get():
                self.lineout_button.toggle()
                self.initialize_plot()
                return
            file_decimal = self.open_file(file_path)
            try:
                self.data = np.array(pd.read_table(file_path, decimal=file_decimal, skiprows=self.skip_rows(file_path), skip_blank_lines=True, dtype=np.float64))
            except: 
                self.data = np.loadtxt(file_path)
            self.first_plot = self.make_line_plot("data", "ax1")
        
        self.plot_axis_parameters("ax1")

        self.update_plot(None)
        self.plot_counter += 1


    def plot_axis_parameters(self, axis):
        axis = getattr(self, axis)
        # if the hide_ticks option has been chosen in the multiplot, settings window
        if self.hide_ticks.get():
            axis.axes.xaxis.set_ticks([])
            axis.axes.yaxis.set_ticks([])

        if self.use_grid_lines.get():
            axis.grid(visible=True, which=self.grid_ticks, axis=self.grid_axis)
            axis.minorticks_on()
        else:
            axis.grid(visible=False)

        # place labels only on the left and bottom plots (if there are multiplots)
        if self.uselabels_button.get() == 1:
            if self.plot_counter > self.cols*(self.rows-1): 
                axis.set_xlabel(self.ent_xlabel.get())
            if (self.plot_counter-1) % self.cols == 0: 
                axis.set_ylabel(self.ent_ylabel.get())

        
    def make_line_plot(self, dat, ax):
        data = getattr(self,dat)
        axis = getattr(self,ax)

        if len(self.data[0, :]) == 1:
            self.data = np.vstack((range(len(self.data)), self.data[:, 0])).T
        
        # create dictionary of the plot key word arguments
        self.plot_kwargs = dict(
            linestyle = self.linestyle,
            marker = self.markers,
            linewidth = self.linewidth,
            )
        
        if self.uselabels_button.get(): 
            self.plot_kwargs["label"] = self.ent_legend.get()

        if self.uselims_button.get():
            self.ylim_slider.set([np.min(data[:, 1]), max(self.ymax,np.max(data[:,1]))])

        if self.normalize_button.get(): self.normalize()
        ######### create the plot
        plot = getattr(axis, self.plot_type)
        plot_object, = plot(data[:, 0], moving_average(data[:, 1], self.moving_average), **self.plot_kwargs)
        #########
        if self.uselabels_button.get() and self.ent_legend.get() != "": axis.legend()

        # create the list
        self.listnames["self.my_listname{}".format(self.plot_counter)] = App.create_label(self.tabview.tab("Data Table"), width=100, text=self.optmenu.get(),sticky='w', row=0, column=self.plot_counter, pady=(0,0), padx=10)
        self.lists["self.my_frame{}".format(self.plot_counter)] = App.create_table(self.tabview.tab("Data Table"), data=data, width=300, sticky='ns', row=1, column=self.plot_counter, pady=(20,10), padx=10)
        for widget_key, widget in self.lists.items():
            widget.configure(width=min(300,0.75*self.tabview.winfo_width()/(1.1*self.plot_counter+1)))

        # create the fit
        if self.use_fit == 1 and not(self.FFT_button.get() and ax=="ax1"):
            self.make_fit_plot(ax, dat)

        if self.FFT_button.get() and ax == "ax1":
            self.create_FFT_border_lines(ax)

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
        
        self.fit_lineout, = axis.plot(data[minimum:maximum, 0], self.fit_plot(self.function, self.params, data[minimum:maximum,:]), **self.plot_kwargs)
        if self.display_fit_params_in_plot.get():
            axis.text(0.05, 0.9, FitWindow.get_fitted_labels(self.fit_window), ha='left', va='top', transform=axis.transAxes, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.6'), size=8)

    def make_image_plot(self, file_path):
        print(file_path)
        self.data = cv2.imdecode(np.fromfile(file_path, np.uint8), cv2.IMREAD_UNCHANGED)
        print(self.data.shape)
        # if the image has color channels, make sure they have the right order, if all channels are the same, reduce it to one channel
        if len(self.data.shape) == 3:
            self.data = cv2.cvtColor(self.data, cv2.COLOR_BGR2RGB)

            b,g,r = self.data[:,:,0], self.data[:,:,1], self.data[:,:,2]
            if (b==g).all() and (b==r).all():
                self.data = self.data[...,0]
        print(self.data.shape)
        # create dictionary of the plot key word arguments
        self.plot_kwargs = dict(
            cmap = self.cmap_imshow,
            interpolation = self.interpolation,
            aspect = self.aspect
            )

        if self.convert_pixels.get() == True:
            self.ax1.set(xticks=np.arange(0,len(self.data[0,:]), self.label_dist/self.pixel_size), xticklabels=np.arange(0,len(self.data[0,:])*self.pixel_size, self.label_dist))
            self.ax1.set(yticks=np.arange(0,len(self.data[:,0]), self.label_dist/self.pixel_size), yticklabels=np.arange(0,len(self.data[:,0])*self.pixel_size, self.label_dist))

        # normalize the image brightness
        if self.normalize_button.get(): self.normalize()
        ######### create the plot
        plot = self.ax1.imshow(self.data, **self.plot_kwargs)
        #########
        # use axis labels and add a legend
        if self.uselabels_button.get() and self.ent_legend.get() != "": 
            self.ax1.text(0.95, 0.95, self.ent_legend.get(), ha='right', va='top', transform=self.ax1.transAxes, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

        # use a scalebar on the bottom right corner
        if self.use_scalebar.get():
            scalebar = ScaleBar(self.pixel_size*1e-3, box_color="black", pad=0.4, box_alpha=.4, color="white", location="lower right", length_fraction=0.3, sep=2) # 1 pixel = 0.2 meter
            self.fig.gca().add_artist(scalebar)
        
        # use a colorbar
        if self.use_colorbar.get():
            self.cbar = self.fig.colorbar(plot, fraction=0.045, pad=0.01)
        

    def update_plot(self, val):
        if self.uselims_button.get() == 1: 
            
            self.x_l, self.x_r = self.xlim_slider.get()
            self.y_l, self.y_r = self.ylim_slider.get()

            if isinstance(val, (tuple, float, str, set, dict)):
                for (lim_box, limit) in zip(["xlim_lbox", "xlim_rbox", "ylim_lbox", "ylim_rbox"],
                                            ["x_l", "x_r", "y_l", "y_r"]):
                    box = getattr(self,lim_box)
                    box.delete(0,'end')
                    if self.image_plot:
                        box.insert(0,str(int(getattr(self,limit))))
                    else:
                        box.insert(0,str(round(getattr(self,limit),4-len(str(int(getattr(self,limit)))))))

            if self.image_plot: # display images
                self.ax1.set_xlim(left=float(self.x_l), right=float(self.x_r))
                self.ax1.set_ylim(bottom=float(self.y_l), top=float(self.y_r))

            else: # line plot
                self.ymax = max(self.ymax,np.max(self.data[:,1]))
                if self.normalize_button.get() == 1: self.normalize()
                self.update_limits()

                self.ax1.set_xlim(left=float(  self.x_l - 0.05 * (self.x_r - self.x_l)), right=float(self.x_r + 0.05 * (self.x_r - self.x_l)))
                self.ax1.set_ylim(bottom=float(self.y_l - 0.05 * (self.y_r - self.y_l)), top=float(  self.y_r + 0.05 * (self.y_r - self.y_l)))
                
                if (self.plot_type == "semilogx" or self.plot_type == "loglog"):
                    xlim_l = calc_log_value(self.x_l - 0.05 * (self.x_r - self.x_l), np.min(abs(self.data[:,0])), np.max(self.data[:,0]))
                    xlim_r = calc_log_value(self.x_r + 0.05 * (self.x_r - self.x_l), np.min(abs(self.data[:,0])), np.max(self.data[:,0]))
                    self.ax1.set_xlim(left=float(xlim_l), right=float(xlim_r))
                if (self.plot_type == "semilogy" or self.plot_type == "loglog"):
                    ylim_l = calc_log_value(self.y_l - 0.05 * (self.y_r - self.y_l), np.min(abs(self.data[:,1])), np.max(self.data[:,1]))
                    ylim_r = calc_log_value(self.y_r + 0.05 * (self.y_r - self.y_l), np.min(abs(self.data[:,1])), np.max(self.data[:,1]))
                    self.ax1.set_ylim(bottom=float(ylim_l), top=float(ylim_r))

        if self.FFT_button.get():
            pass
        if self.fit_button.get():
            try:
                self.fit_xmin_line.set_xdata([self.fit_window.fit_borders_slider.get()[0]])
                self.fit_xmax_line.set_xdata([self.fit_window.fit_borders_slider.get()[1]])
            except:
                if self.lineout_button.get() or self.FFT_button.get():
                    self.create_fit_border_lines("ax2")
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
        path = customtkinter.filedialog.askdirectory(initialdir=Standard_path)
        self.folder_path.delete(0, customtkinter.END)
        self.folder_path.insert(0, path)
        global filelist
        filelist = [fname for fname in os.listdir(self.folder_path.get()) if fname.endswith(('.csv', '.dat', '.txt', '.png', '.jpg'))]
        self.optmenu.configure(values=filelist)
        self.optmenu.set(filelist[0])

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
            self.limits_title = App.create_label(self.settings_frame, text="Limits", font=customtkinter.CTkFont(size=16, weight="bold"), row=row, column=1, columnspan=5, padx=20, pady=(20, 5),sticky=None)
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

                # create the entry objects "self.xlim_lbox" ...
                setattr(self, lim_lbox, App.create_entry(self.settings_frame, row=row+i+1, column=1, width=50))
                setattr(self, lim_rbox, App.create_entry(self.settings_frame, row=row+i+1, column=4, width=50))
                setattr(self, lim_slider, App.create_range_slider(self.settings_frame, from_=value_min, to=value_max, command= lambda val=None: self.update_plot(val), row=row+i+1, column =2, width=120, padx=(0,0), columnspan=2, init_value=[value_min, value_max]))

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
        
    def use_labels(self):
        if self.uselabels_button.get() == 1:
            row = 0
            self.settings_frame.grid()
            self.labels_title = App.create_label(self.settings_frame, text="Labels", font=customtkinter.CTkFont(size=16, weight="bold"), row=row, column=1, columnspan=5, padx=20, pady=(20, 5),sticky=None)
            self.ent_xlabel      = App.create_entry(self.settings_frame,column=1, row=row+1, columnspan=2, width=110, placeholder_text="x label")
            self.ent_ylabel      = App.create_entry(self.settings_frame,column=3, row=row+1, columnspan=2, width=110, placeholder_text="y label")
            self.ent_legend      = App.create_entry(self.settings_frame,column=1, row=row+2, columnspan=4, width=240)
            self.ent_label_text = App.create_label(self.settings_frame,column=0, row=row+1, text="x / y")
            self.ent_legend_text = App.create_label(self.settings_frame,column=0, row=row+2, text="legend")
        else:
            for name in ["ent_xlabel","ent_label_text","ent_ylabel","ent_legend","ent_legend_text","labels_title"]:
                getattr(self, name).grid_remove()
            self.close_settings_window()
    
    def multiplot(self):
        self.replot = not self.replot
        if self.multiplot_button.get() == 1:
            if self.lineout_button.get():
                self.rows = 1
            else:
                self.rows = 2
            self.cols = 2

            row = 9
            self.multiplot_title = App.create_label( self.settings_frame,column=1, row=row, text="Multiplot Grid", font=customtkinter.CTkFont(size=16, weight="bold"), columnspan=5, padx=20, pady=(20, 5),sticky=None)
            self.ent_rows        = App.create_entry( self.settings_frame,column=1, row=row+1, width=50, padx=(5,40),columnspan=2)
            self.ent_rows_text   = App.create_label( self.settings_frame,column=0, row=row+1, text="rows")
            self.ent_cols        = App.create_entry( self.settings_frame,column=1, row=row+2, width=50, padx=(5,40),columnspan=2)
            self.ent_cols_text   = App.create_label( self.settings_frame,column=0, row=row+2, text="cols")
            self.switch_ticks    = App.create_switch(self.settings_frame,column=3, row=row+1, text="hide ticks",  command = lambda: self.toggle_boolean(self.hide_ticks), padx=(5,20), columnspan=2, sticky="w")
            self.switch_multiplt = App.create_switch(self.settings_frame,column=3, row=row+2, text="1 multiplot", command = lambda: self.toggle_boolean(self.one_multiplot), padx=(5,20), columnspan=2, sticky="w")
            self.settings_frame.grid()
            self.addplot_button.grid() 
            self.reset_button.grid()
            self.plot_button.grid_remove()

            self.ent_rows.insert(0,str(self.rows))
            self.ent_cols.insert(0,str(self.cols))
            if self.one_multiplot.get(): self.switch_multiplt.select()
            if self.hide_ticks.get(): self.switch_ticks.select()
        else:
            self.plot_button.grid()
            for name in ["ent_rows","ent_rows_text","ent_cols","ent_cols_text","reset_button","addplot_button","multiplot_title","switch_ticks", "switch_multiplt"]:
                getattr(self, name).grid_remove()
            self.close_settings_window()
            # reset number of rows and cols for plotting to single plot value
            self.rows = 1
            self.cols = 1
        self.initialize_plot()

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
            self.lineout_xy = " x-line "
            self.settings_frame.grid()
            row = 12
            self.lineout_title    = App.create_label( self.settings_frame,column=1, row=row, text="Lineout", font=customtkinter.CTkFont(size=16, weight="bold"), columnspan=5, padx=20, pady=(20, 5),sticky=None)
            self.choose_ax_button = App.create_segmented_button(self.settings_frame, column = 1, row=row+2, values=[" x-line ", " y-line "], command=self.initialize_lineout, columnspan=2, padx=(10,10), sticky="w")
            self.line_entry       = App.create_entry(self.settings_frame, row=row+1, column=1, width=50)
            self.line_slider      = App.create_slider(self.settings_frame, from_=0, to=max(len(self.data[:,0]), len(self.data[0,:])), command= lambda val=None: self.plot_lineout(val), row=row+1, column =2, width=170, padx=(0,10), columnspan=3)
            self.ent_line_text    = App.create_label( self.settings_frame,column=0, row=row+1, text="line")
            self.save_lineout_button = App.create_button(self.settings_frame, column = 3, row=row+2, text="Save Lineout", command= lambda: self.save_data_file(self.lineout_data), width=80, columnspan=2)
            self.line_entry.insert(0,str(int(len(self.data[:,0])/2)))
            self.line_entry.bind("<KeyRelease>", lambda event, val=self.line_entry, slider_widget=self.line_slider: (slider_widget.set(float(val.get())), self.plot_lineout(val)))
            self.choose_ax_button.set("x-line")
            self.choose_ax_button.configure(border_width= 5, fg_color = ["#3a7ebf", "#1f538d"], unselected_color = ["#3a7ebf", "#1f538d"], unselected_hover_color=["#325882", "#14375e"])
            self.initialize_lineout(" x-line ")
        else:
            try:
                for name in ["lineout_title", "line_entry", "line_slider", "ent_line_text", "choose_ax_button", "save_lineout_button"]:
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
            self.x_lineout_max = len(self.data[0,:])
            self.y_lineout_min = 0
            self.y_lineout_max = len(self.data[:,0])

        asp_image = len(self.data[self.y_lineout_min:self.y_lineout_max,0]) / len(self.data[0,self.x_lineout_min:self.x_lineout_max])

        if val == " x-line ":
            if self.line_slider.get() >= self.y_lineout_max: self.line_slider.set(self.y_lineout_max-1)
            self.lineout_data = np.array([self.data[int(self.line_slider.get()),self.x_lineout_min:self.x_lineout_max]]).T
            self.axline = self.ax1.axhline([int(self.line_slider.get())])
            self.line_slider.configure(from_=self.y_lineout_min, to=self.y_lineout_max-1)
            self.lineout_xy = " x-line "
        elif val == " y-line ":
            if self.line_slider.get() >= self.x_lineout_max: self.line_slider.set(self.x_lineout_max-1)
            self.lineout_data = np.array([self.data[self.y_lineout_min:self.y_lineout_max,int(self.line_slider.get())]]).T
            self.axline = self.ax1.axvline([int(self.line_slider.get())])
            self.line_slider.configure(from_=self.x_lineout_min, to=self.x_lineout_max-1)
            self.lineout_xy = " y-line "
        
        # reset slider value when the lineout is changed and the value is out of bounds
        self.line_slider.set(self.line_slider.get())
        self.line_entry.delete(0, 'end')
        self.line_entry.insert(0,str(int(self.line_slider.get())))
        
        self.ax2 = self.fig.add_subplot(1,2,2)
        max_value = np.max(self.data)
        
        scale = 1
        if self.convert_pixels.get():
            scale = self.pixel_size

        self.lineout_data = np.vstack((np.arange(0,len(self.lineout_data)*scale,scale), self.lineout_data[:, 0])).T
        self.lineout_plot = self.make_line_plot("lineout_data", "ax2")
        self.plot_axis_parameters("ax2")

        # asp = np.diff(self.ax2.get_xlim())[0] / np.diff(self.ax2.get_ylim())[0]*asp_image
        asp = np.diff(self.ax2.get_xlim())[0] / max_value*asp_image

        self.ax2.set_aspect(abs(asp))
        self.ax2.set_ylim(0, max_value)

        self.canvas.draw()
        self.choose_ax_button.set(None)

    def plot_lineout(self, val):
        self.line_entry.delete(0, 'end')
        self.line_entry.insert(0,str(int(self.line_slider.get())))
        
        if self.lineout_xy == " x-line ":
            self.lineout_data[:,1] = np.array([self.data[int(self.line_slider.get()),self.x_lineout_min:self.x_lineout_max]])
            self.axline.set_ydata([int(self.line_slider.get())])
        elif self.lineout_xy == " y-line ":
            self.lineout_data[:,1] = np.array([self.data[self.y_lineout_min:self.y_lineout_max,int(self.line_slider.get())]])
            self.axline.set_xdata([int(self.line_slider.get())])
        
        self.lineout_plot.set_ydata(self.lineout_data[:,1])
        if self.use_fit == 1:
            minimum = np.argmin(abs(self.lineout_data[:,0] - self.fit_window.fit_borders_slider.get()[0]))
            maximum = np.argmin(abs(self.lineout_data[:,0] - self.fit_window.fit_borders_slider.get()[1]))
            self.fit_lineout.set_xdata(self.lineout_data[minimum:maximum,0])
            self.fit_lineout.set_ydata(self.fit_plot(self.function, self.params, self.lineout_data[minimum:maximum,:]))
        self.canvas.draw()

    def fourier_trafo(self):
        # if self.image_plot == True:
        #     self.FFT_button.deselect()

        if self.FFT_button.get():
            self.display_fit_params_in_plot.set(False)
            if self.multiplot_button.get():
                self.multiplot_button.toggle()
            if self.lineout_button.get():
                self.lineout_button.toggle()
            self.replot = True
            self.one_multiplot.set(False)
            self.multiplot_button.configure(state="disabled")
            self.lineout_button.configure(state="disabled")
            

            self.cols = 2
            self.rows = 1
            self.settings_frame.grid()
            row = 15
            self.FFT_title    = App.create_label( self.settings_frame,column=1, row=row, text="Fourier Transform", font=customtkinter.CTkFont(size=16, weight="bold"), columnspan=5, padx=20, pady=(20, 5),sticky=None)
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
        self.ax2 = self.fig.add_subplot(1,2,2)

        self.FFT_data = self.Fourier_transform("data")
        self.FFT_plot = self.make_line_plot("FFT_data", "ax2")
        # self.FFT_plot, = self.ax2.plot(self.FFT_data[:,0],self.FFT_data[:,1], **self.plot_kwargs)

        # Determine the width of the displayed FT window
        argmax = np.argmax(self.FFT_data[:,1])
        width = np.max(np.argwhere(self.FFT_data[:,1] > 1e-3*np.max(self.FFT_data[:,1]))) - argmax + int(0.01*len(self.FFT_data[:,1])/(self.padd_zeros_slider.get()+1))
        # width = abs(np.argmax(self.FFT_data[:,1]) - np.argmin(abs(self.FFT_data[:,1]-0.5*np.max(self.FFT_data[:,1]))))
        # self.ax2.set_xlim(self.FFT_data[argmax-10*half_width,0], self.FFT_data[argmax+10*half_width,0])
        self.ax2.set_xlim(self.FFT_data[argmax-width, 0], self.FFT_data[argmax+width, 0])

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
        np.savetxt(file_name, data, fmt="%.4e")
        
    def update_limits(self):
        if self.image_plot:
            self.xlim_slider.configure(from_=0, to=len(self.data[0,:]))
            self.ylim_slider.configure(from_=0, to=len(self.data[:,0]))
        else:
            self.xlim_slider.configure(from_=np.floor(np.min(self.data[:, 0])), to=np.ceil(np.max(self.data[:, 0])))  
            self.ylim_slider.configure(from_=np.min(self.data[:, 1]), to=self.ymax)

        
    def normalize_setup(self):
        if self.normalize_button.get():
            if self.uselims_button.get() and not self.image_plot:
                self.update_limits()
                self.ylim_slider.set([np.min(self.data[:, 1]),np.max(self.data[:, 1])])
        self.initialize()
        
    
    def normalize(self):
        if self.image_plot:
            self.data = self.data * 0.5/np.mean(self.data)*self.enhance_value
            self.data[self.data>1] = 1
        elif self.change_norm_to_area.get() == False:
            self.data[:, 1] /= np.max(self.data[:, 1])
        elif self.change_norm_to_area.get() == True:
            self.data[:, 1] /= (np.sum(self.data[:, 1])*abs(self.data[0,0]-self.data[1,0]))

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
        if (self.image_plot and self.save_plain_image.get()):
            if self.uselims_button.get(): 
                plt.imsave(file_name, self.data[int(self.y_l):int(self.y_r),int(self.x_l):int(self.x_r)], cmap=self.cmap_imshow)
            else: 
                plt.imsave(file_name, self.data, cmap=self.cmap_imshow)
        else:
            self.fig.savefig(file_name, bbox_inches='tight')

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
        else:
            self.color = "#1a1a1a"
        if self.initialize_plot_has_been_called: 
            self.toolbar.pack_forget()
            self.toolbar = self.create_toolbar()

    def toggle_boolean(self, boolean):
        boolean.set(not boolean.get())
        self.initialize_plot()

    def close_settings_window(self):
        if not (self.uselabels_button.get() or self.uselims_button.get() or self.fit_button.get() or self.multiplot_button.get() or self.lineout_button.get()):
            self.settings_frame.grid_remove()
            
    def open_toplevel(self,cls):
        if self.toplevel_window is None or not self.toplevel_window.winfo_exists():
            self.toplevel_window = cls(self)  # create window if its None or destroyed
            self.toplevel_window.focus()  # focus it
        else:
            self.toplevel_window.focus()  # if window exists focus it
            
    def open_widget(self,cls):
        if self.fit_button.get() == 1:
            self.fit_window = cls(self)  # create window if its None or destroyed
        else:
            self.fit_window.close_window()
            self.close_settings_window()
            self.fit_window.destroy()
            
    def on_closing(self):
        quit() # Python 3.12 works
        self.destroy() # Python 3.9 on Laptop
        
        
"""
############################################################################################
##############################    Settings Window    #######################################
############################################################################################

"""

class SettingsWindow(customtkinter.CTkToplevel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry("600x550")
        self.title("Settings")
        self.app = app

        # values for image plots
        self.cmap_imshow = ['magma','hot','viridis', 'plasma', 'inferno', 'cividis', 'gray']
        self.aspect_values = ['equal', 'auto']
        
        # values for line plots
        self.values = {'solid': '-', 'dashed': '--', 'dotted': ':', 'dashdotted': '-.', 'no line': ''}
        self.markers = {'none':'', 'point':'.', 'circle':'o', 'pixel':',', 'triangle down': 'v', 'triangle up': '^', 'x': 'x', 'diamond': 'D'}
        self.grid_lines = ['major','minor','both']
        self.grid_axis = ['both', 'x', 'y']
        self.cmap = ['tab10', 'tab20', 'tab20b','tab20c', 'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3']
        self.moving_average = ['1','2','4','6','8','10','16']
        self.plot_type = {'Normal': 'plot', 'Semi Logarithmic x': 'semilogx', 'Semi Logarithmic y': 'semilogy', 'Log-Log plot': 'loglog'}

        App.create_label(self, column=1, row=0, columnspan=2, text="Image plot settings", font=customtkinter.CTkFont(size=16, weight="bold"),padx=20, pady=(20, 5), sticky=None)
        App.create_label(self, column=1, row=6, columnspan=2, text="Line plot settings", font=customtkinter.CTkFont(size=16, weight="bold"),padx=20, pady=(20, 5), sticky=None)
        # values for image plots
        self.pixel_size         = App.create_entry(   self, column=1, row=1, columnspan=2, text="Pixel size [µm]", placeholder_text="7.4 µm", sticky='w')
        self.label_dist         = App.create_entry(   self, column=1, row=2, columnspan=2, text="Label step [mm]", placeholder_text="1 mm", sticky='w')
        self.aspect             = App.create_optionMenu(self, column=1, row=3, columnspan=2, values=self.aspect_values, text="Aspect",  command=self.set_plot_settings)
        self.cmap_imshow_list   = App.create_optionMenu(self, column=1, row=4, columnspan=2, values=self.cmap_imshow,                text="Colormap",command=self.set_plot_settings)
        self.enhance_slider     = App.create_slider(  self, column=1, row=5, columnspan=2, init_value=1, from_=0.1, to=2, 
                                                    command= lambda value, var="enhance_var": self.update_slider_value(value, var), text="Enhance value", number_of_steps=100)

        # values for line plots
        self.plot_type_list     = App.create_optionMenu(self, column=1, row=7, columnspan=2, values=list(self.plot_type.keys()),text="Plot type",command=self.set_plot_settings)
        self.linestyle_list     = App.create_optionMenu(self, column=1, row=8, columnspan=2, values=list(self.values.keys()), text="Line style", command=self.set_plot_settings)
        self.marker_list        = App.create_optionMenu(self, column=1, row=9, columnspan=2, values=list(self.markers.keys()),text="Marker",     command=self.set_plot_settings)
        self.cmap_list          = App.create_optionMenu(self, column=1, row=10, columnspan=2, values=self.cmap,                text="Colormap",   command=self.set_plot_settings)
        self.grid_lines_list    = App.create_optionMenu(self, column=1, row=11, width=90, values=self.grid_lines, text="Grid",  command=self.set_plot_settings)
        self.grid_axis_list     = App.create_optionMenu(self, column=2, row=11, width=90, values=self.grid_axis, command=self.set_plot_settings)
        self.moving_av_list     = App.create_optionMenu(self, column=1, row=12, columnspan=2, values=self.moving_average,      text="Average",   command=self.set_plot_settings)
        self.linewidth_slider   = App.create_slider(  self,  column=1, row=13, columnspan=2, init_value=1, from_=0.1, to=2, 
                                                    command= lambda value, var="lw_var": self.update_slider_value(value, var), text="line width", number_of_steps=100)

        self.reset_button           = App.create_button(self, column=4, row=0, text="Reset Settings",   command=self.reset_values, width=130, pady=(20,5))
        self.scale_switch_button    = App.create_switch(self, column=4, row=1, text="Use Scalebar",     command=lambda: self.toggle_boolean(self.app.use_scalebar))
        self.cbar_switch_button     = App.create_switch(self, column=4, row=2, text="Use Colorbar",     command=lambda: self.toggle_boolean(self.app.use_colorbar))
        self.convert_pixels_button  = App.create_switch(self, column=4, row=3, text="Convert Pixels",   command=lambda: self.toggle_boolean(self.app.convert_pixels))
        self.save_plain_img_button  = App.create_switch(self, column=4, row=4, text="Save plain images",command=lambda: self.toggle_boolean(self.app.save_plain_image))
        self.hide_params_button     = App.create_switch(self, column=4, row=5, text="Hide fit params",  command=lambda: self.toggle_boolean(self.app.display_fit_params_in_plot))
        self.norm_switch_button     = App.create_switch(self, column=4, row=6, text="Normalize 'Area'", command=lambda: self.toggle_boolean(self.app.change_norm_to_area))
        self.grid_lines_button     = App.create_switch(self, column=4, row=7, text="Use Grid",         command=lambda: self.toggle_boolean(self.app.use_grid_lines))
        
        # Slider labels    
        self.enhance_var = customtkinter.StringVar()  # StringVar to hold the label value
        self.lw_var = customtkinter.StringVar()  # StringVar to hold the label value
        App.create_label(self, textvariable=self.enhance_var, column=3, row=5, width=30, anchor='w', sticky='w')
        App.create_label(self, textvariable=self.lw_var, column=3, row=13, width=30, anchor='w', sticky='w')
        
        self.pixel_size.bind("<KeyRelease>", self.set_plot_settings)
        self.label_dist.bind("<KeyRelease>", self.set_plot_settings)

        #set initial values
        self.init_values()
        
    # initiate current values
    def init_values(self):
        self.pixel_size.insert(0,str(self.app.pixel_size*1e3))
        self.label_dist.insert(0,str(self.app.label_dist))
        self.aspect.set(self.app.aspect)
        self.cmap_imshow_list.set(self.app.cmap_imshow)
        self.linestyle_list.set(list(self.values.keys())[list(self.values.values()).index(self.app.linestyle)])
        self.marker_list.set(list(self.markers.keys())[list(self.markers.values()).index(self.app.markers)])
        self.plot_type_list.set(list(self.plot_type.keys())[list(self.plot_type.values()).index(self.app.plot_type)])
        self.cmap_list.set(self.app.cmap)
        self.grid_lines_list.set(self.app.grid_ticks)
        self.grid_axis_list.set(self.app.grid_axis)
        self.moving_av_list.set(self.app.moving_average)
        self.enhance_slider.set(self.app.enhance_value)
        self.linewidth_slider.set(self.app.linewidth)
        self.update_slider_value(self.app.enhance_value, "enhance_var")
        self.update_slider_value(self.app.linewidth, "lw_var")

        if self.app.use_scalebar.get() == True: self.scale_switch_button.select()
        if self.app.use_colorbar.get() == True: self.cbar_switch_button.select()
        if self.app.convert_pixels.get() == True: self.convert_pixels_button.select()
        if self.app.display_fit_params_in_plot.get() == False: self.hide_params_button.select()
        if self.app.change_norm_to_area.get() == True: self.norm_switch_button.select()
        if self.app.save_plain_image.get(): self.save_plain_img_button.select()
        if self.app.use_grid_lines.get(): self.grid_lines_button.select()

    # reset to defaul values
    def reset_values(self):
        # Here we define the standard values!!!
        self.app.pixel_size = 7.4e-3
        self.app.label_dist = 1
        self.pixel_size.delete(0, 'end')
        self.pixel_size.insert(0,str(self.app.pixel_size*1e3))
        self.label_dist.delete(0, 'end')
        self.label_dist.insert(0,str(self.app.label_dist))
        self.aspect.set(next(iter(self.aspect_values)))
        self.cmap_imshow_list.set(self.cmap_imshow[0])

        self.linestyle_list.set(next(iter(self.values)))
        self.marker_list.set(next(iter(self.markers)))
        self.cmap_list.set(self.cmap[0])
        self.grid_lines_button.deselect()
        self.grid_lines_list.set(self.grid_lines[0])
        self.grid_axis_list.set(self.grid_axis[0])
        self.moving_av_list.set(str(self.moving_average[0]))
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
            self.app.pixel_size=float(self.pixel_size.get())*1e-3
            self.app.label_dist=float(self.label_dist.get())
        except: pass 
        self.app.aspect=self.aspect.get()
        self.app.cmap_imshow=self.cmap_imshow_list.get()
        self.app.enhance_value=self.enhance_slider.get()

        self.app.linestyle=self.values[self.linestyle_list.get()]
        self.app.markers=self.markers[self.marker_list.get()]
        self.app.cmap=self.cmap_list.get()
        self.app.linewidth=self.linewidth_slider.get()
        self.app.moving_average = int(self.moving_av_list.get())
        self.app.grid_ticks = self.grid_lines_list.get()
        self.app.grid_axis = self.grid_axis_list.get()
        self.app.plot_type=self.plot_type[self.plot_type_list.get()]

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
        self.labels_title = App.create_label(app.settings_frame, text="Fit Function", font=customtkinter.CTkFont(size=16, weight="bold"),row=self.row , column=1, columnspan=5, padx=20, pady=(20, 5),sticky=None)
        self.function = {'Gaussian': gauss, 'Lorentz': lorentz, 'Linear': linear, 'Quadratic': quadratic, 'Exponential': exponential, 'Square Root': sqrt}
        self.function_label = App.create_label(app.settings_frame, text="", column=1, row=self.row +1, width=80, columnspan=4, anchor='e', sticky="w")
        self.function_label_list = {'Gaussian': "f(x) = a·exp(-(x-b)²/(2·c²)) + d", 
                                    'Lorentz': "f(x) = a/[(x²-b²)² + c²·b²]",
                                    'Linear': "f(x) = a·x + b", 
                                    'Quadratic': "f(x) = a·x² + b·x + c", 
                                    'Exponential': "f(x) = a·exp(b·x) + c", 
                                    'Square Root': "f(x) = a·sqrt(b·x) + c"}
        self.function_list_label = App.create_label(app.settings_frame,text="Fit", column=0,row=self.row +2, sticky="e")
        self.function_list = App.create_combobox(app.settings_frame, values=list(self.function.keys()), command=self.create_params, width=110, column=1, row=self.row +2, columnspan=2, sticky="w")
        self.function_list.set('Gaussian')

        if app.lineout_button.get():
            data = app.lineout_data
        else:
            data = app.data

        if (not app.image_plot or app.lineout_button.get()):
            self.fit_borders_slider = App.create_range_slider(app.settings_frame, from_=np.min(data[:,0]), to=np.max(data[:,0]), command= lambda val=None: app.update_plot(val), row=self.row+2, column =3, width=120, padx=(0,0), columnspan=2, init_value=[np.min(data[:,0]), np.max(data[:,0])])

        self.params = []
        self.error = []
        self.create_params('Gaussian')
        # self.set_fit_params()
        app.settings_frame.grid()
        
    def create_params(self, function_name):
        # set the label of the function, e.g f(x) = a*x+b
        self.function_name = function_name
        self.function_label.configure(text = self.function_label_list[self.function_list.get()])
        
        #Determine the number of arguments of the used function
        self.number_of_args = len(signature(self.function[function_name]).parameters) - 1
        
        for i,name in enumerate(["a","b","c","d"]):
            self.create_parameter(name, i)

        if self.function_name == "Gaussian":
            self.widget_dict['str_var_FWHM'] = customtkinter.StringVar()
            # self.widget_dict['fitted_FWHM'] = App.create_label(app.settings_frame,textvariable=self.widget_dict['str_var_FWHM'], column=2,row=row + 7, sticky="e", padx=(0,20), columnspan=2)
            # self.widget_dict['fittedlabel_FWHM']  = App.create_label(app.settings_frame,text='FWHM =', column=1,row=row + 7, sticky=None, columnspan=1)
        # else:
            # try: self.widget_dict['fitted_FWHM'].grid_remove(), self.widget_dict['fittedlabel_FWHM'].grid_remove()
            # except: 
                # pass
        self.set_fit_params(app.data)
        self.row += 2 + self.number_of_args
    
    def create_parameter(self, name, arg_number):
        # try deleting the parameters of previous fit function
        try: 
            for attribute in ["fit", "fitlabel", "fitted"]:
                self.widget_dict[f'{attribute}_{name}'].grid_remove()
        except: pass
        
        if arg_number < self.number_of_args:
            self.widget_dict[f'fit_{name}'], self.widget_dict[f'fitlabel_{name}'] = self.create_fit_entry(label_text=name, column=1, row=self.row + arg_number + 3)
            self.widget_dict[f'str_var_{name}'] = customtkinter.StringVar() # StringVar to hold the label value
            self.widget_dict[f'fitted_{name}'] = App.create_label(app.settings_frame, textvariable=self.widget_dict[f'str_var_{name}'], column=3, width=50, row=self.row + arg_number + 3, padx=(0, 20), columnspan=2)
        
    def create_fit_entry(self, label_text, column, row):
        fit_variable = App.create_entry(app.settings_frame, column=column, row=row, width=80, placeholder_text="0", sticky='w', columnspan=2)
        fit_label    = App.create_label(app.settings_frame, text=label_text, column=column-1, row=row, anchor='e') 
        return fit_variable, fit_label
    
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
        string = self.function_label_list[self.function_list.get()] 
        
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
                if (self.function_name == "Gaussian" or self.function_name == "Lorentz") and i == 1:
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

    # def insert_data(self, values):
        # if len(values) != len(self.columns):
        #     raise ValueError(f"Number of values {len(values)} must match the number of columns {len(self.columns)}")

        # self.text_widget.insert("end", "\t".join(map(str, values)) + "\n")

        # self.update()
        
if __name__ == "__main__":
    app = App()
    app.state('zoomed')
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()