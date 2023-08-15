# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 14:28:26 2022

@author: Martin
"""
import tkinter
import os
import customtkinter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import ctypes
import io
import pandas as pd
import matplotlib
from cycler import cycler
from scipy.optimize import curve_fit as cf
from inspect import signature # get number of arguments of a function

version_number = "23/08"
plt.style.use('default')
matplotlib.rc('font', family='serif')
matplotlib.rc('font', serif='Times New Roman')

ctypes.windll.shcore.SetProcessDpiAwareness(1)

Standard_path = os.path.dirname(os.path.abspath(__file__))
filelist = [fname for fname in os.listdir(Standard_path) if fname.endswith(('.csv', '.dat', '.txt'))]

def exponential(x, a, b, c):
    return a * np.exp(b * x) + c

def gauss(x, a, b, c, d):
    return a * np.exp(-(x - b)**2 / (2 * c**2)) + d

def sqrt(x, a, b, c):
    return a * np.sqrt(b * x) + c

def linear(x, a, b):
    return np.polyval([a,b], x)

def quadratic(x, a, b, c):
    return np.polyval([a,b,c], x)

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        customtkinter.set_appearance_mode("dark")
        customtkinter.set_default_color_theme("dark-blue")
        self.title("Graph interactive editor for smooth visualization GIES v."+version_number)
        self.geometry("1100x600")
        self.replot = False
        self.initialize_ui()
        self.initialize_plot_has_been_called = False
        self.folder_path.insert(0, Standard_path)
        self.toplevel_window = None
        # hide the multiplot button
        self.addplot_button.grid_remove() 
        self.plot2_button.grid_remove()
        self.initialize_variables()
        
        
    def initialize_variables(self):
        self.fit_gauss = 0
        self.row = 5
        self.ymax = 0
        self.linestyle='-'
        self.markers=''
        self.cmap="tab10"
        self.linewidth=1
        self.use_fit = 0
        self.function = gauss
        self.params = [1,1030,10,0]
        self.fitted_params = [0,0,0,0]
        self.color = "#1a1a1a"
        self.normalize_type = "maximum"
        self.ax1 = None
        self.plot_counter = 0
        
        
    # user interface, gets called when the program starts 
    def initialize_ui(self):

        self.sidebar_frame = customtkinter.CTkFrame(self, width=200, height=600, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=999, sticky="nesw")
        self.sidebar_frame.rowconfigure(12, weight=1)
        self.rowconfigure(11,weight=1)

        App.create_label(self.sidebar_frame, text="GIES v."+version_number, font=customtkinter.CTkFont(size=20, weight="bold"),width=20, row=0, column=0, padx=20, pady=(20, 10))
        
        self.tabview = customtkinter.CTkTabview(self, width=250)
        self.tabview.grid(row=11, column=1, padx=(20, 0), pady=(20, 0), columnspan=2, sticky="nsew")
        self.tabview.add("Show Plots")
        self.tabview.add("Data Table")
        # self.tabview.rowconfigure(0, weight=1)  # Make the plot area expandable vertically
        # self.tabview.columnconfigure(0, weight=1) 
        
        #buttons
        frame = self.sidebar_frame
        
        self.plot_button        = App.create_button(frame, text="Plot graph",             command=self.initialize_plot, column=0, row=5, width=200, pady=(15,5))
        self.plot2_button       = App.create_button(frame, text="Reset",                  command=self.initialize_plot, column=0, row=5, width=90, padx = (10,120), pady=(15,5))   # Multiplot button
        self.addplot_button     = App.create_button(frame, text="Multiplot",              command=self.plot,            column=0, row=5, width=90, padx = (120,10), pady=(15,5))   # Multiplot button
        self.load_button        = App.create_button(frame, text="\U0001F4C1 Load Folder", command=self.read_file_list,  column=0, row=1, width=200)
        self.save_button        = App.create_button(frame, text="\U0001F4BE Save Figure", command=self.save_figure,     column=0, row=3, width=200)
        self.settings1_button   = App.create_button(frame, text="Plot settings",          command =lambda: self.open_toplevel(SettingsWindow), column=0, row=4, width=90, padx = (10,120))
        self.settings2_button   = App.create_button(frame, text="Fit settings",           command =lambda: self.open_toplevel(FitWindow), column=0, row=4, width=90, padx = (120,10))
        self.prev_button        = App.create_button(frame, text="<<",                     command=lambda: self.change_plot(self.replot,-1), column=0, row=6, width=90, padx = (10,120))
        self.next_button        = App.create_button(frame, text=">>",                     command=lambda: self.change_plot(self.replot,1), column=0, row=6, width=90, padx = (120,10))
        
        #switches
        self.multiplot_button   = App.create_switch(frame, text="Multiplot", command=self.multiplot,  column=0, row=7, width=200, sticky='w', padx=20, pady=(10,5))
        self.uselabels_button   = App.create_switch(frame, text="Use labels", command=self.use_labels,  column=0, row=8, width=200, sticky='w', padx=20)
        self.uselims_button     = App.create_switch(frame, text="Use limits", command=self.use_limits,  column=0, row=9, width=200, sticky='w', padx=20)
        self.normalize_button   = App.create_switch(frame, text="Normalize", command=self.normalize_setup,    column=0, row=10, width=200, sticky='w', padx=20)
        
        
        self.settings_frame = customtkinter.CTkFrame(self, width=1, height=600, corner_radius=0)
        self.settings_frame.grid(row=0, column=4, rowspan=999, sticky="nesw")
        self.columnconfigure(2,weight=1)
        # self.columnconfigure(1,weight=1)
        
        self.appearance_mode_label = App.create_label(self.sidebar_frame, text="Appearance Mode:", width=20, row=13, column=0, padx=20, pady=(10, 0), sticky="w")
        self.appearance_mode_optionemenu = App.create_optionMenu(self.sidebar_frame, values=["Dark","Light", "System"], command=self.change_appearance_mode_event, width=200, row=14, column=0, padx=20, pady=(5,10))
        
        #entries
        self.folder_path = self.create_entry(column=2, row=0, text="Folder path", columnspan=2, width=600, padx=10, pady=10, sticky="w")
        
        #dropdown menu
        self.optmenu = self.create_combobox(values=filelist, text="File name",row=1, column=2, columnspan=2, width=600, sticky="w")
        if not filelist == []: self.optmenu.set(filelist[0])
        
        #table
        # 


    def create_label(self, width, row, column, text=None, anchor='e', sticky=None, textvariable=None, padx=None, pady=None, font=None, columnspan=1, **kwargs):
        label = customtkinter.CTkLabel(self, text=text, textvariable=textvariable, width=width, anchor=anchor, font=font,  **kwargs)
        label.grid(row=row, column=column, sticky=sticky, columnspan=columnspan, padx=padx, pady=pady, **kwargs)
        return label

    def create_entry(self, width, row, column, text=None, columnspan=1, padx=10, pady=5, placeholder_text=None, sticky="n", **kwargs):
        entry = customtkinter.CTkEntry(self, width, placeholder_text=placeholder_text, **kwargs)
        entry.grid(row=row, column=column, columnspan=columnspan, padx=padx, pady=pady, sticky=sticky, **kwargs)
        if text is not None:
            App.create_label(self, text=text, column=column-1, row=row, width=80, anchor='e', sticky='e')
        return entry

    def create_button(self, text, command, width, row, column, columnspan=1, padx=10, pady=5, sticky=None, **kwargs):
        button = customtkinter.CTkButton(self, text=text, command=command, width=width, **kwargs)
        button.grid(row=row, column=column, columnspan=columnspan, padx=padx, pady=pady, sticky=sticky, **kwargs)
        return button

    def create_switch(self, text, command, width, row, column, columnspan=1, padx=10, pady=5, sticky=None, **kwargs):
        switch = customtkinter.CTkSwitch(self, text=text, command=command, **kwargs)
        switch.grid(row=row, column=column, columnspan=columnspan, padx=padx, pady=pady, sticky=sticky, **kwargs)
        return switch
    
    def create_combobox(self, values, width, column, row, state='readonly', command=None, text=None, columnspan=1, padx=10, pady=5, sticky=None, **kwargs):
        combobox = customtkinter.CTkComboBox(self, values=values, command=command, state=state, width=width, **kwargs)
        combobox.grid(column=column, row=row, columnspan=columnspan, padx=padx, pady=pady, sticky=sticky, **kwargs)
        if text is not None:
            App.create_label(self, text=text, column=column-1, row=row, width=80, anchor='e',pady=pady, sticky='e')
        return combobox
    
    def create_optionMenu(self, values, column, row, command=None, text=None, width=None, columnspan=1, padx=10, pady=5, sticky=None, **kwargs):
        combobox = customtkinter.CTkOptionMenu(self, values=values, width=width, command=command, **kwargs)
        combobox.grid(column=column, row=row, columnspan=columnspan, padx=padx, pady=pady, sticky=sticky, **kwargs)
        if text is not None:
            App.create_label(self, text=text, column=column-1, row=row, width=20, anchor='e', pady=pady, sticky='e')
        return combobox
    
    def create_slider(self, from_, to, width, row, column, text=None, init_value=None, command=None, columnspan=1, number_of_steps=1000, padx = 10, pady=5, **kwargs):
        slider = customtkinter.CTkSlider(self, from_=from_, to=to, width=width, command=command)
        slider.grid(column=column, row=row, columnspan=columnspan, padx=padx, pady=pady, **kwargs)
        if text is not None:
            App.create_label(self, text=text, column=column-1, row=row, width=80, anchor='e', sticky='e')

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
    

    def initialize_plot(self):
        
        # Clear the previous plot content
        if self.ax1 is not None:
            self.ax1.clear()  
            self.canvas.get_tk_widget().destroy()
            self.toolbar.destroy()
            for widget_key, widget in self.lists.items():
                widget.destroy()
            self.plot_counter = 0
            plt.close(self.fig)
            
        
        self.lists = {}
        if not self.initialize_plot_has_been_called:
            self.initialize_plot_has_been_called = True
            
        self.fig, self.ax1 = plt.subplots(1,1,figsize=(15,4), constrained_layout=True, dpi=150)
        self.ax1.set_prop_cycle(cycler(color=plt.get_cmap(self.cmap).colors))
        if self.uselabels_button.get() == 1:
            self.ax1.set_xlabel(self.ent_xlabel.get())
            self.ax1.set_ylabel(self.ent_ylabel.get())
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tabview.tab("Show Plots"))
        # self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=2,padx=20, pady=(15,0), sticky='nsew')
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar = self.create_toolbar()
        self.ymax = 0
        self.plot()

    def create_toolbar(self):
        toolbar_frame = customtkinter.CTkFrame(master=self.tabview.tab("Show Plots"))
        # toolbar_frame.grid(row=12, column=1, columnspan=3)
        toolbar_frame.pack()
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.config(background=self.color)
        toolbar._message_label.config(background=self.color, foreground="white")
        toolbar.winfo_children()[-2].config(background=self.color)
        toolbar.update()
        return toolbar_frame
        

    def plot(self):
        if not self.initialize_plot_has_been_called:
            self.initialize_plot()
        file_path = os.path.join(self.folder_path.get(), self.optmenu.get())
        file_decimal = self.open_file(file_path)
        self.data = np.array(pd.read_table(file_path, decimal=file_decimal, skiprows=self.skip_rows(file_path), skip_blank_lines=True))

        if len(self.data[0, :]) == 1:
            self.data = np.vstack((range(len(self.data)), self.data[:, 0])).T

        if self.normalize_button.get() == 1: self.normalize()
        
        # create dictionary of the plot key word arguments
        plot_kwargs = dict(
            linestyle = self.linestyle,
            marker = self.markers,
            linewidth = self.linewidth,
            )
        
        if self.uselabels_button.get() == 1: 
            plot_kwargs["label"] = self.ent_legend.get()
            self.ax1.legend()
            
        if self.uselims_button.get() == 1:
            self.ylim_l.set(np.min(self.data[:, 1]))
            self.ylim_r.set(max(self.ymax,np.max(self.data[:,1])))
            
        # create the plot
        self.ax1.plot(self.data[:, 0], self.data[:, 1], **plot_kwargs)
        
        # create the list
        self.lists["self.my_frame{}".format(self.plot_counter)] = App.create_table(self.tabview.tab("Data Table"), data=self.data, width=300, sticky='ns', row=0, column=self.plot_counter, pady=(20,10), padx=10)
        for widget_key, widget in self.lists.items():
            widget.configure(width=min(300,0.75*self.tabview.winfo_width()/(1.1*self.plot_counter+1)))
        
        # create the fit
        if self.use_fit == 1:
            self.ax1.plot(self.data[:, 0], self.fit_plot(self.function, self.params), **plot_kwargs)
        self.update_plot()
        self.plot_counter += 1
        # self.fig.tight_layout()
        

    def update_plot(self):
        self.ymax = max(self.ymax,np.max(self.data[:,1]))
        if self.normalize_button.get() == 1: self.normalize()

        if self.uselims_button.get() == 1:
            self.update_limits()
            self.ax1.set_xlim(left=float(self.xlim_l.get()  - 0.05 * (self.xlim_r.get() - self.xlim_l.get())),
                              right=float(self.xlim_r.get() + 0.05 * (self.xlim_r.get() - self.xlim_l.get())))
            self.ax1.set_ylim(bottom=float(self.ylim_l.get()- 0.05 * (self.ylim_r.get() - self.ylim_l.get())),
                              top=float(self.ylim_r.get()   + 0.05 * (self.ylim_r.get() - self.ylim_l.get())))

        self.canvas.draw()

    def change_plot(self, replot, direction):
        index = filelist.index(self.optmenu.get()) + direction
        if 0 <= index < len(filelist):
            self.optmenu.set(filelist[index])
            if not replot: self.initialize_plot()
            if replot: self.plot()

    def fit_plot(self, function, initial_params):
        params,K=cf(function,self.data[:, 0],self.data[:, 1], p0=initial_params)
        #write fit parameters in an array to access
        FitWindow.set_fitted_values(self.toplevel_window, params, K)
        return function(self.data[:,0], *params)
        
    def read_file_list(self):
        path = customtkinter.filedialog.askdirectory(initialdir=Standard_path)
        self.folder_path.delete(0, customtkinter.END)
        self.folder_path.insert(0, path)
        global filelist
        filelist = [fname for fname in os.listdir(self.folder_path.get()) if fname.endswith(('.csv', '.dat', '.txt'))]
        self.optmenu.configure(values=filelist)
        self.optmenu.set(filelist[0])

    def use_limits(self):
        if not self.initialize_plot_has_been_called: return
        if self.uselims_button.get() == 1:
            
            self.settings_frame.grid()
            self.limits_title = App.create_label(self.settings_frame, text="Limits", font=customtkinter.CTkFont(size=16, weight="bold"),width=20, row=self.row-1, column=0, columnspan=2, padx=20, pady=(20, 10))
            self.labx = App.create_label(self.settings_frame, text="x limits", column=0, row=self.row, width=20, sticky='e', padx=(10,5))
            self.laby = App.create_label(self.settings_frame, text="y limits", column=0, row=self.row+2, width=20, sticky='e', padx=(10,5))

            # Slider
            self.xlim_l = App.create_slider(self.settings_frame, from_=np.min(self.data[:,0]), to=np.max(self.data[:,0]), width=150, 
                                             command=lambda val: self.update_plot(), row=self.row, column =1, sticky='w', padx=(5,15))
            self.xlim_r = App.create_slider(self.settings_frame, from_=np.min(self.data[:,0]), to=np.max(self.data[:,0]), width=150, 
                                             command=lambda val: self.update_plot(), row=self.row+1, column =1, sticky='w', padx=(5,15))
            self.ylim_l = App.create_slider(self.settings_frame, from_=np.min(self.data[:,1]), to=np.max(self.ymax), width=150, 
                                             command=lambda val: self.update_plot(), row=self.row+2, column =1, sticky='w', padx=(5,15))
            self.ylim_r = App.create_slider(self.settings_frame, from_=np.min(self.data[:,1]), to=np.max(self.ymax), width=150, 
                                             command=lambda val: self.update_plot(), row=self.row+3, column =1, sticky='w', padx=(5,15))

            self.xlim_l.set(np.min(self.data[:, 0]))
            self.xlim_r.set(np.max(self.data[:, 0]))
            self.ylim_l.set(np.min(self.data[:, 1]))
            self.ylim_r.set(self.ymax)
            self.row += 5
            # self.columnconfigure(4,weight=1)
        else:
            for name in ["xlim_l","xlim_r","ylim_l","ylim_r","labx","laby","limits_title"]:
                getattr(self, name).grid_remove()
                
            if self.uselabels_button.get() == 0:
                self.settings_frame.grid_remove()

            self.row -= 5
        self.update_plot()
        
    def use_labels(self):
        if self.uselabels_button.get() == 1:
            self.settings_frame.grid()
            self.labels_title = App.create_label(self.settings_frame, text="Labels", font=customtkinter.CTkFont(size=16, weight="bold"),width=20, row=self.row-1, column=0, columnspan=2, padx=20, pady=(20, 10))
            self.ent_xlabel      = App.create_entry(self.settings_frame,column=1, row=self.row, width=150, sticky='w')
            self.ent_xlabel_text = App.create_label(self.settings_frame,column=0, text="x label", row=self.row, width=20,padx=(10,5))
            self.ent_ylabel      = App.create_entry(self.settings_frame,column=1, row=self.row+1, width=150, sticky='w')
            self.ent_ylabel_text = App.create_label(self.settings_frame,column=0, text="y label", row=self.row+1, width=20,padx=(10,5))
            self.ent_legend      = App.create_entry(self.settings_frame,column=1, row=self.row+2, width=150, sticky='w')
            self.ent_legend_text = App.create_label(self.settings_frame,text="legend", width=20, row=self.row+2, column=0,padx=(10,5))
            self.row += 4 
        else:
            for name in ["ent_xlabel","ent_xlabel_text","ent_ylabel","ent_ylabel_text","ent_legend","ent_legend_text","labels_title"]:
                getattr(self, name).grid_remove()
            if self.uselims_button.get() == 0:
                self.settings_frame.grid_remove()
            self.row -= 4
    
    def multiplot(self):
        self.replot = not self.replot
        if self.multiplot_button.get() == 1:
            self.addplot_button.grid() 
            self.plot2_button.grid()
            self.plot_button.grid_remove()
        else:
            self.addplot_button.grid_remove() 
            self.plot2_button.grid_remove()
            self.plot_button.grid()
            
        
    def update_limits(self):
        self.xlim_l.configure(from_=np.floor(np.min(self.data[:, 0])), to=np.ceil(np.max(self.data[:, 0])))
        self.xlim_r.configure(from_=np.floor(np.min(self.data[:, 0])), to=np.ceil(np.max(self.data[:, 0])))
        self.ylim_l.configure(from_=np.min(self.data[:, 1]), to=self.ymax)
        self.ylim_r.configure(from_=np.min(self.data[:, 1]), to=self.ymax)

        
    def normalize_setup(self):
        if self.normalize_button.get() == 1:
            self.initialize_plot()
            if self.uselims_button.get() == 1:
                self.update_limits()
                self.ylim_l.set(np.min(self.data[:, 1]))
                self.ylim_r.set(np.max(self.data[:, 1]))
        self.initialize_plot()
    
    def normalize(self):
        if self.normalize_type == "maximum":
            self.data[:, 1] /= np.max(self.data[:, 1])
        if self.normalize_type == "area":
            self.data[:, 1] /= (np.sum(self.data[:, 1])*abs(self.data[0,0]-self.data[1,0]))

        self.ymax = max([self.ymax, np.max(self.data[:, 1])])
            
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
            self.create_toolbar()

            
    def open_toplevel(self,cls):
        if self.toplevel_window is None or not self.toplevel_window.winfo_exists():
            self.toplevel_window = cls(self)  # create window if its None or destroyed
            self.toplevel_window.focus()  # focus it
        else:
            self.toplevel_window.focus()  # if window exists focus it
            
            
    def on_closing(self):
        self.destroy()

class UseLabels(customtkinter.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.app = app
        # add widgets onto the frame, for example:
        self.label = customtkinter.CTkLabel(self)
        self.label.grid(row=0, column=0, padx=20)
        
# neues Fenster
class SettingsWindow(customtkinter.CTkToplevel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry("500x300")
        self.title("Settings")
        self.app = app
        self.values = {'solid': '-', 'dashed': '--', 'dotted': ':', 'dashdotted': '-.', 'no line': ''}
        self.markers = {'none':'', 'point':'.', 'circle':'o', 'pixel':',', 'triangle down': 'v', 'triangle up': '^', 'x': 'x', 'diamond': 'D'}
        self.cmap = ['tab10', 'tab20', 'tab20b','tab20c', 'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
                      'Set1', 'Set2', 'Set3']
        
        # command funktioniert doch, muss noch implementiert werden für set_plot settings
        self.linestyle_list     = App.create_combobox(self, values=list(self.values.keys()), text="line style", 
                                                      command=self.set_plot_settings,   width=200, column=1, row=0, pady=(20,5))
        self.marker_list        = App.create_combobox(self, values=list(self.markers.keys()),text="marker",     
                                                      command=self.set_plot_settings,   width=200, column=1, row=2)
        self.cmap_list          = App.create_combobox(self, values=self.cmap,                text="colormap",   
                                                      command=self.set_plot_settings,   width=200, column=1, row=3)
        self.reset_button       = App.create_button(self, text="Reset Settings",command=self.reset_values, width=130, column=3, row=0, pady=(20,5))
        self.linewidth_slider   = App.create_slider(self, init_value=1, from_=0.1, to=2, command=self.update_slider_value, text="line width", width=200, row=1, column=1, number_of_steps=10)
        self.norm_switch_button = App.create_switch(self, text="Normalize 'Area'", command=self.change_normalize_function,  column=3, row=2, width=200)
        
        #labels        
        self.lw_var = customtkinter.StringVar()  # StringVar to hold the label value
        App.create_label(self, textvariable=self.lw_var, column=2, row=1, width=30, anchor='w', sticky='w')
        
        #set initial values
        self.linestyle_list.set(list(self.values.keys())[list(self.values.values()).index(self.app.linestyle)])
        self.marker_list.set(list(self.markers.keys())[list(self.markers.values()).index(self.app.markers)])
        self.cmap_list.set(self.app.cmap)
        self.linewidth_slider.set(self.app.linewidth)
        self.update_slider_value(self.app.linewidth)
        if self.app.normalize_type == "area": self.light_theme_button.select()
   
    def reset_values(self):
        self.linestyle_list.set(next(iter(self.values)))
        self.marker_list.set(next(iter(self.markers)))
        self.cmap_list.set(self.cmap[0])
        self.linewidth_slider.set(1) # before next line!
        self.update_slider_value(1)
        
        
    def update_slider_value(self, value):
        self.lw_var.set(str(round(value,2)))  # Update the StringVar
        self.set_plot_settings(None)
        
    def set_plot_settings(self, val):
        self.app.linestyle=self.values[self.linestyle_list.get()]
        self.app.markers=self.markers[self.marker_list.get()]
        self.app.cmap=self.cmap_list.get()
        self.app.linewidth=self.linewidth_slider.get()

    
    def change_normalize_function(self):
        if self.norm_switch_button.get() == 1:
            self.app.normalize_type = "area"
        else:
            self.app.normalize_type = "maximum"

# neues Fenster
class FitWindow(customtkinter.CTkToplevel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry("500x300")
        self.title("Fit Function")
        self.app = app
        self.fit_switch = App.create_switch(self, text="Fit Function", command=self.set_plot_settings, width=200, row=0, column=1)
        self.function = {'Gaussian': gauss, 'Linear': linear, 'Quadratic': quadratic, 'Exponential': exponential, 'Square Root': sqrt}
        self.function_label = App.create_label(self, text="", column=2, row=0, width=80, columnspan=2, anchor='w', sticky='w')
        self.function_label_list = {'Gaussian': "f(x) = a * exp(-(x - b)^2 / (2 * c^2)) + d", 
                                    'Linear': "f(x) = a * x + b", 
                                    'Quadratic': "f(x) = a * x^2 + b * x + c", 
                                    'Exponential': "f(x) = a * exp(b * x) + c", 
                                    'Square Root': "f(x) = a * sqrt(b * x) + c"}
        
        self.function_list = App.create_combobox(self, values=list(self.function.keys()), command=self.create_params, text="Fit Function",   width=130, column=1, row=1)
        
        
    def create_params(self, function_name):
        # set the label of the function, e.g f(x) = a*x+b
        self.function_name = function_name
        self.function_label.configure(text = self.function_label_list[self.function_list.get()])
        
        #Determine the number of arguments of the used function
        self.number_of_args = len(signature(self.function[function_name]).parameters) - 1
        
        for i,name in enumerate(["a","b","c","d"]):
            self.create_parameter(name, i)

        if self.function_name == "Gaussian":
            self.str_var_FWHM = customtkinter.StringVar()
            self.fitted_FWHM = App.create_label(self,textvariable=self.str_var_FWHM, width=100, column=3,row=6, sticky="e", padx=(5,10))
            self.fittedlabel_FWHM  = App.create_label(self,text='FWHM =', column=2,width=80, row=6)
        else:
            try: self.fitted_FWHM.grid_remove(), self.fittedlabel_FWHM.grid_remove()
            except: pass
        self.set_plot_settings()
    
    def create_parameter(self, name, arg_number):
        # exec ... executes a string as python code
        try: 
            for attribute in ["fit_", "fitlabel_", "fitted:", "fittedlabel_"]:
                getattr(self, attribute + name).grid_remove()
        except: pass
        
        if arg_number < self.number_of_args:
            exec("self.fit_"+name+", self.fitlabel_"+name+"  = self.create_fit_entry(label_text='"+name+"', column=1, row="+str(arg_number+2)+")")
            exec("self.fittedlabel_"+name+"  = App.create_label(self,text='fit "+name+" =', column=2,width=80, row="+str(arg_number+2)+")")
            exec("self.str_var_"+name+" = customtkinter.StringVar()")  # StringVar to hold the label value
            
            exec("self.fitted_"+name+"  = App.create_label(self,textvariable=self.str_var_"+name+", column=3,width=50,sticky='e', row="+str(arg_number+2)+", padx=(5,10))")
        
    def create_fit_entry(self, label_text, column, row):
        fit_variable = App.create_entry(self, column=column, row=row, width=130, placeholder_text="0")
        fit_label    = App.create_label(self, text=label_text, column=column-1, row=row, width=80, anchor='e', sticky='e')
        return fit_variable, fit_label
    
    def set_fitted_values(self, params, error):
        for i,name in enumerate(["a","b","c","d","FWHM"]):
            if self.function_name == "Gaussian":
                self.number_of_args += 1
                params = np.append(params, 2*np.sqrt(2*np.log(2))*params[2]) 
                error = np.pad(error, ((0, 1), (0, 1)), mode='constant')
                error[-1,-1] = error[2,2] * params[-1]
            
            if i >= self.number_of_args: break   # take only the right number of arguments
            print(params[i])
            getattr(self, "str_var_" + name).set(str(round(params[i],2))+" ± "+str(round(np.sqrt(error[i,i]),2)))
    
    def set_plot_settings(self):
        self.app.params = []
        for i,name in enumerate(['a','b','c','d']):
            if i >= self.number_of_args: break   # take only the right number of arguments
            try:
                value = float(getattr(self, "fit_" + name).get())
                self.app.params.append(value)
            except ValueError:
                self.app.params.append(0)

        self.app.function = self.function[self.function_list.get()]
        self.app.use_fit = self.fit_switch.get()

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
