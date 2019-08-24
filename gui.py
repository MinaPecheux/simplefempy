# Copyright 2019 - M. Pecheux
# [SimpleFEMPy] A basic Python PDE solver with the finite elements method
# ------------------------------------------------------------------------------
# gui.py - Graphical interface for the SimpleFEMPy (Tkinter application)
# ==============================================================================
import sys, os, io, re
if sys.version_info[0] < 3:
    import Tkinter as tk
    import tkFileDialog as filedialog
else:
    import tkinter as tk
    import tkinter.filedialog as filedialog

from simplefempy.settings import AUTHOR, DATE, VERSION, LIB_SETTINGS, TK_INSTANCE
LIB_SETTINGS['using_tk'] = True

from simplefempy.discretizor import DiscreteDomain
from simplefempy.solver import VariationalFormulation
from simplefempy.utils.logger import Logger
from simplefempy.utils.maths import FunctionalFunction, dirac
from simplefempy.converter import save_to_csv, save_to_vtk

import numpy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def labeled_input(canvas, label):
    """Creates a Tkinter Entry with a description label.
    
    Parameters
    ----------
    canvas : Tkinter widget
        Parent for the new widget.
    label : str
        Text of the description label.
    """
    widget = tk.PanedWindow(canvas, orient=tk.HORIZONTAL)
    widget.pack(side=tk.TOP, fill=tk.X, expand='yes')
    label = tk.Label(widget, text=label + ':')
    widget.add(label)
    entry_var = tk.StringVar() 
    entry_var.set('')
    entry = tk.Entry(widget, textvariable=entry_var)
    widget.add(entry)
    widget.pack(fill=tk.X, expand='yes')
    return entry_var

def labeled_spinbox(canvas, label, **kwargs):
    """Creates a Tkinter Spinbox with a description label.
    
    Parameters
    ----------
    canvas : Tkinter widget
        Parent for the new widget.
    label : str
        Text of the description label.
    """
    widget = tk.PanedWindow(canvas, orient=tk.HORIZONTAL)
    label = tk.Label(widget, text=label + ':')
    widget.add(label)
    spinbox = tk.Spinbox(widget, **kwargs)
    widget.add(spinbox)
    widget.pack(fill=tk.X, expand='yes')
    return spinbox

def checkbox(canvas, label, **kwargs):
    """Creates a Tkinter Checkbutton with a description label.
    
    Parameters
    ----------
    canvas : Tkinter widget
        Parent for the new widget.
    label : str
        Text of the description label.
    """
    check_var = tk.IntVar()
    btn = tk.Checkbutton(canvas, text=label, variable=check_var, **kwargs)
    canvas.add(btn)
    return check_var

def dropdown(canvas, choices, **kwargs):
    """Creates a Tkinter dropdown list.
    
    Parameters
    ----------
    canvas : Tkinter widget
        Parent for the new widget.
    choices : list(str)
        List of choices in the dropdown (the first one is selected by default).
    """
    var = tk.StringVar()
    var.set(choices[0]) # set default option
    menu = tk.OptionMenu(canvas, var, *choices, **kwargs)
    canvas.add(menu)
    return var

class SFEMPyStream(io.StringIO):
    
    """Custom stream for the SimpleFEMPy GUI application."""
    
    def __init__(self, app, *args, **kwargs):
        self.app = app
        io.StringIO.__init__(self, *args, **kwargs)
        
    def write(self, s):
        # transmit to app interface
        if s != '\n': self.app.log(s)

class SFEMPyApp(tk.Tk):
    
    """Custom Tkinter GUI application that provides a SimpleFEMPy interface."""
    
    TOP_MENU_HEIGHT = 45
    PRIMITIVES = [ 'Line', 'Rectangle', 'Circle', 'Ring' ]
    CMAPS = [ 'coolwarm', 'viridis', 'gnuplot', 'gnuplot2', 'plasma',
              'spring', 'summer', 'autumn', 'winter', 'cool',
              'hot', 'copper', 'bone', 'magma', 'jet' ]
    
    def __init__(self, *args, **kwargs):
        w = kwargs.pop('width')
        h = kwargs.pop('height')
        
        tk.Tk.__init__(self, *args, **kwargs)
        self.title('SimpleFEMPy (GUI Interface)')
        self.protocol("WM_DELETE_WINDOW", self.exit)
        
        # global canvas
        canvas = tk.Canvas(self, width=w, height=h)
        canvas.pack()
        
        # top menu frame
        top_menu = tk.PanedWindow(canvas, width=w, height=SFEMPyApp.TOP_MENU_HEIGHT,
                                       bg='#dddddd', orient=tk.HORIZONTAL)
        top_menu.add(tk.Label(top_menu,
            text='SimpleFEMPy (v. {}), {} - {}'.format(VERSION, AUTHOR, DATE)))
        self.logger = tk.Text(top_menu, bg='black', fg='white',
                              height=SFEMPyApp.TOP_MENU_HEIGHT)
        top_menu.add(self.logger)
        top_menu.add(tk.Button(top_menu, text='Load', command=self.load_session))
        top_menu.add(tk.Button(top_menu, text='Save', command=self.save_session))
        top_menu.add(tk.Button(top_menu, text='Clear', command=self.reset_session))
        top_menu.add(tk.Button(top_menu, text='Quit', command=self.exit))
        top_menu.pack()
        
        # main container frame
        container = tk.PanedWindow(canvas, width=w, orient=tk.HORIZONTAL,
                    height=(h-SFEMPyApp.TOP_MENU_HEIGHT), bg='#bbbbdd')
        container.pack(side=tk.TOP, fill='both', expand='yes')
        # .. left panel: input window
        input_panel = tk.Frame(container, padx=20, pady=20)
        input_panel_title = tk.Label(input_panel,
                                text='Domain and Problem Settings', width=100)
        input_panel_title.pack()

        # .... domain info
        input_panel_domain = tk.LabelFrame(input_panel, text='Domain', padx=10, pady=10)
        input_panel_domain.pack(fill='both', expand='yes')
        
        self.discrete_domain_name = labeled_input(input_panel_domain, 'Domain Name')
        self.discrete_domain_name.set('Vh')

        domain_columns = tk.PanedWindow(input_panel_domain, orient=tk.HORIZONTAL)
        domain_columns.pack(fill=tk.X)
        
        domain_primitives = tk.Canvas(domain_columns)
        self.domain_primitives_list = tk.Listbox(domain_primitives)
        self.domain_primitives_list.bind('<<ListboxSelect>>', self.on_primitive_select)
        for i, prim in enumerate(SFEMPyApp.PRIMITIVES):
            self.domain_primitives_list.insert(i+1, prim)
        self.domain_primitives_list.pack()
        self.domain_primitives_borders = labeled_spinbox(domain_primitives, 'Nb borders', from_=1, to=4)
        domain_prim_load_btn = tk.Button(domain_primitives, text='Load Primitive',
                                         command=self.load_primitive)
        domain_prim_load_btn.pack()
        domain_columns.add(domain_primitives)
        
        domain_col2 = tk.PanedWindow(domain_columns, orient=tk.VERTICAL)
        self.domain_params = tk.PanedWindow(domain_col2, orient=tk.VERTICAL, height=200)
        domain_col2.add(self.domain_params)

        domain_load_btn = tk.Button(domain_col2, text='Load Mesh', command=self.load_mesh)
        domain_col2.add(domain_load_btn)
        
        domain_columns.add(domain_col2)

        # .... variational formulation info
        input_panel_varf = tk.LabelFrame(input_panel,
                            text='Variational Formulation', padx=10, pady=10)
        input_panel_varf.pack(side=tk.TOP, fill='both', expand='yes')
        
        self.varf_str = labeled_input(input_panel_varf, 'Problem Equations\n(FreeFem++ format)')
        
        varf_vars_top = tk.PanedWindow(input_panel_varf, orient=tk.HORIZONTAL)
        varf_vars_top.add(tk.Label(varf_vars_top, text='Variables'))
        varf_vars_top.add(tk.Button(varf_vars_top, text='Add Variable', command=self.add_variable))
        varf_vars_top.pack(side=tk.TOP)
        
        self.varf_vars_list = tk.PanedWindow(input_panel_varf, orient=tk.VERTICAL)
        self.varf_vars_list.pack(fill=tk.X, expand='yes')
        
        varf_btns = tk.PanedWindow(input_panel_varf, orient=tk.HORIZONTAL)
        varf_solve_btn = tk.Button(varf_btns, text='Solve', command=self.solve_problem)
        varf_btns.add(varf_solve_btn)
        varf_sol_btn = tk.Button(varf_btns, text='Show Solution', command=self.set_figure)
        varf_btns.add(varf_sol_btn)
        varf_mask_btn = tk.Button(varf_btns, text='Show Domain', command=self.show_domain)
        varf_btns.add(varf_mask_btn)
        varf_btns.pack()

        container.add(input_panel)

        # .. right panel: output window
        output_panel = tk.Frame(container)
        output_panel_title = tk.Label(output_panel, text='Output', width=100)
        output_panel_title.pack()
        
        output_plots = tk.PanedWindow(output_panel, orient=tk.HORIZONTAL)
        output_plots.pack()
        self.output_type = tk.IntVar()
        output_plots.add(tk.Radiobutton(output_plots, text='Plot Solution',
            variable=self.output_type, value=1, command=self.set_output_type))
        output_plots.add(tk.Radiobutton(output_plots, text='Plot Custom',
            variable=self.output_type, value=2, command=self.set_output_type))
        self.output_custom_str = tk.StringVar()
        self.output_custom = tk.Entry(output_plots, textvariable=self.output_custom_str)
        output_plots.add(self.output_custom)
        self.output_type.set(1)
        self.output_custom.config(state='disabled')

        self.output_canvas = tk.Canvas(output_panel, width=w//2, height=h//2)
        self.output_canvas.pack(fill=tk.Y, expand='yes')
        
        output_commands = tk.PanedWindow(output_panel, orient=tk.HORIZONTAL)
        output_commands.pack()
        self.output_opt_3d = checkbox(output_commands, '3D View',
                                      command=self.set_figure)
        self.output_opt_value = checkbox(output_commands, 'Show Value',
                                         command=self.set_figure)
        self.output_opt_labels = checkbox(output_commands, 'Show Labels',
                                         command=self.set_figure)
        self.output_opt_triangulate = checkbox(output_commands, 'Triangulate',
                                         command=self.set_figure)
        self.output_opt_cmap = dropdown(output_commands, SFEMPyApp.CMAPS,
                                        command=self.set_cmap)
        output_commands.add(tk.Button(output_commands, text='Export Solution',
                                      command=self.export_solution))

        container.add(output_panel)
        
        container.paneconfig(input_panel, minsize=w//3)
        container.paneconfig(output_panel, minsize=w//2)
        
        # prepare useful variables
        self.discrete_domain      = None
        self.domain_reference     = None
        self.domain_cur_primitive = None
        self.solution             = None
        self.solution_variables   = {}
        self.plot                 = 'solution'
        self.fig, self.fig_photo  = None, None
        self.vars                 = []

    def _parse_custom_str(self):
        """Parses the custom output string into the corresponding computed
        result."""
        regex   = r'(?P<sign>[+|-])?(?P<coeff>[^+-]*\*)?(?P<func>[^+-]*)'
        matches = re.finditer(regex, self.output_custom_str.get())
        sol     = numpy.zeros_like(self.solution)
        for m in matches:
            if len(m.group(0)) == 0: continue
            sign = -1. if m.group('sign') == '-' else 1.
            if m.group('coeff') is not None:
                coeff = int(m.group('coeff')[:-1]) * sign
            else:
                coeff = sign
            if m.group('func') == 'solution': s = self.solution
            else:
                try:
                    func = self.solution_variables[m.group('func')]
                except KeyError:
                    Logger.slog('ERROR: function "{}" is not '
                                'defined!'.format(m.group('func')))
                    return None
                s = func(*self.discrete_domain.points)
            sol = sol + coeff * s
        return sol
        
    def _parse_vars_str(self):
        """Parses the variable inputs into the corresponding variables."""
        regex    = r'(?P<name>[^\s]*)\s*=\s*(?P<value>[^\n;]*)'
        cur_vars = '; '.join([v.get() for v in self.vars])
        matches  = re.finditer(regex, cur_vars)
        vars     = { m.group('name'): eval(m.group('value')) for m in matches }
        return vars
        
    def exit(self):
        """Custom exit function to properly destroy the Tkinter application."""
        self.quit()
        self.destroy()
        
    def log(self, s):
        """Logs information to the window-included app console.
        
        Parameters
        ----------
        s : str
            String to log.
        """
        self.logger.delete('1.0', tk.END)
        self.logger.insert(tk.END, s)
        
    def add_variable(self):
        """Adds a variable empty field to the interface."""
        entry_var = tk.StringVar() 
        entry_var.set('')
        entry = tk.Entry(self.varf_vars_list, textvariable=entry_var)
        self.varf_vars_list.add(entry)
        self.vars.append(entry_var)
        
    def on_primitive_select(self, event):
        """Callback function for the primitives listbox item selection.
        
        Parameters
        ----------
        event : Tkinter event
            Listbox selection event.
        """
        old_value = self.domain_cur_primitive
        w = event.widget
        index = int(w.curselection()[0])
        value = w.get(index)
        if old_value is None or value != old_value:
            self.set_primitive_params(value)
        
    def set_primitive_params(self, value):
        """Sets the necessary parameters fields depending on the selected
        primitive.
        
        Parameters
        ----------
        value : str
            Currently selected primitive name.
        """
        # remove previous widgets
        for child in self.domain_params.winfo_children(): child.destroy()
        # replace according to type of primitive
        if value == 'Line':
            self.domain_param1 = labeled_spinbox(self.domain_params, 'Length', from_=0.5, to=100, format='%.2f')
            self.domain_param2 = labeled_spinbox(self.domain_params, 'Step', from_=2, to=100)
            self.domain_param1.delete(0, tk.END)
            self.domain_param1.insert(0, '1.0')
            self.domain_param2.delete(0, tk.END)
            self.domain_param2.insert(0, '10')
        elif value == 'Rectangle':
            self.domain_param1 = labeled_spinbox(self.domain_params, 'X-Size', from_=0.5, to=100, format='%.2f')
            self.domain_param2 = labeled_spinbox(self.domain_params, 'Y-Size', from_=0.5, to=100, format='%.2f')
            self.domain_param3 = labeled_spinbox(self.domain_params, 'Step', from_=2, to=100)
            self.domain_param1.delete(0, tk.END)
            self.domain_param1.insert(0, '1.0')
            self.domain_param2.delete(0, tk.END)
            self.domain_param2.insert(0, '1.0')
            self.domain_param3.delete(0, tk.END)
            self.domain_param3.insert(0, '10')
        elif value == 'Circle':
            self.domain_param1 = labeled_spinbox(self.domain_params, 'Radius', from_=0.5, to=100, format='%.2f')
            self.domain_param2 = labeled_spinbox(self.domain_params, 'Angle Step', from_=4, to=100)
            self.domain_param3 = labeled_spinbox(self.domain_params, 'Radius Step', from_=2, to=100)
            self.domain_param1.delete(0, tk.END)
            self.domain_param1.insert(0, '1.0')
            self.domain_param2.delete(0, tk.END)
            self.domain_param2.insert(0, '30')
            self.domain_param3.delete(0, tk.END)
            self.domain_param3.insert(0, '10')
        elif value == 'Ring':
            self.domain_param1 = labeled_spinbox(self.domain_params, 'Inner Radius', from_=0.5, to=100, format='%.2f')
            self.domain_param2 = labeled_spinbox(self.domain_params, 'Outer Radius', from_=1, to=100, format='%.2f')
            self.domain_param3 = labeled_spinbox(self.domain_params, 'Angle Step', from_=2, to=100)
            self.domain_param4 = labeled_spinbox(self.domain_params, 'Radius Step', from_=2, to=100)
            self.domain_param3.delete(0, tk.END)
            self.domain_param3.insert(0, '30')
            self.domain_param4.delete(0, tk.END)
            self.domain_param4.insert(0, '10')
    
        self.domain_cur_primitive = value
        
    def load_primitive(self):
        """Creates a DiscreteDomain instance with a primitive reference."""
        selected = self.domain_cur_primitive
        if selected is None: return

        b = int(self.domain_primitives_borders.get())
        if selected == 'Line':
            length = float(self.domain_param1.get())
            step   = int(self.domain_param2.get())
            self.discrete_domain = DiscreteDomain.line(length, step=step,
                                                       nb_borders=b)
        elif selected == 'Rectangle':
            xsize = float(self.domain_param1.get())
            ysize = float(self.domain_param2.get())
            step  = int(self.domain_param3.get())
            self.discrete_domain = DiscreteDomain.rectangle(xsize, ysize,
                                                    step=step, nb_borders=b)
        elif selected == 'Circle':
            radius = float(self.domain_param1.get())
            astep  = int(self.domain_param2.get())
            rstep  = int(self.domain_param3.get())
            self.discrete_domain = DiscreteDomain.circle(radius, astep=astep,
                                                    rstep=rstep, nb_borders=b)
        elif selected == 'Ring':
            iradius = float(self.domain_param1.get())
            oradius = float(self.domain_param2.get())
            if iradius > oradius:
                Logger.slog('ERROR: cannot initialize "Ring" primitive with '
                            'Inner Radius > Outer Radius.')
                return
            astep = int(self.domain_param3.get())
            rstep = int(self.domain_param4.get())
            self.discrete_domain = DiscreteDomain.ring(iradius, oradius,
                                        astep=astep, rstep=rstep, nb_borders=b)
        self.reset_problem()
        self.domain_reference = '_Primitive_'
        self.set_figure()
        
    def load_mesh(self):
        """Creates a DiscreteDomain instance with a file reference."""
        self.reset_problem()
        file_reader = filedialog.askopenfilename(title='Choose a mesh file:')
        if len(file_reader) > 0:
            # if necessary, reset primitive-linked variables
            for child in self.domain_params.winfo_children(): child.destroy()
            self.domain_cur_primitive = None
            # read file into a new instance
            self.domain_reference = file_reader
            self.discrete_domain = DiscreteDomain.from_file(file_reader)
            self.set_figure()
        
    def solve_problem(self):
        """Solves the Variational Formulation."""
        # check for missing information
        varf_str = self.varf_str.get()
        if varf_str == '': return
        if self.discrete_domain is None: return
        # get local variables and additional problem variables
        vars = locals()
        discrete_domain_name = self.discrete_domain_name.get()
        if discrete_domain_name == '': return
        vars[discrete_domain_name] = self.discrete_domain
        self.solution_variables = self._parse_vars_str()
        vars.update(self.solution_variables)
        # create and solve weak formulation
        varf = VariationalFormulation.from_str(varf_str, vars)
        self.solution = varf.solve()
        # reupdate the output figure
        self.set_figure()
        
    def reset_problem(self):
        """Resets the Variational Formulation."""
        self.solution = None
        self.solution_variables = {}
        
    def reset_session(self):
        """Loads a SimpleFEMPy session (domain, variables, weak formulation and
        output settings)."""
        self.discrete_domain_name.set('Vh')
        self.discrete_domain      = None
        self.domain_reference     = None
        self.domain_cur_primitive = None
        if self.fig is not None: self.fig.clear()
        self.fig                  = None
        # if necessary, destroy previous figure
        if self.fig_photo is not None:
            self.fig_photo.get_tk_widget().pack_forget()
            self.fig_photo = None
        self.domain_primitives_list.selection_clear(0, tk.END)
        for child in self.domain_params.winfo_children(): child.destroy()
        self.domain_primitives_borders.delete(0, tk.END)
        self.domain_primitives_borders.insert(0, '1')
        
        self.reset_problem()
        for child in self.varf_vars_list.winfo_children(): child.destroy()
        self.vars = []
        self.varf_str.set('')
        
        self.plot = 'solution'
        self.output_type.set(1)
        self.output_custom_str.set('')
        self.output_opt_value.set(0)
        self.output_opt_labels.set(0)
        self.output_opt_triangulate.set(0)
        self.output_opt_3d.set(0)
        self.output_opt_cmap.set(SFEMPyApp.CMAPS[0])
        Logger.slog('Cleared session.', stackoffset=100)
        
    def show_domain(self):
        """Shows the plain domain (with no solution)."""
        self.set_figure(only_domain=True)
        
    def set_cmap(self, *args):
        """Wrapper to avoid wrong parameter transmission to the figure update
        function."""
        self.set_figure()
        
    def set_output_type(self):
        """Sets the type of output depending on the radio button value:
        - 1 is the solution
        - 2 is a custom output
        """
        if self.output_type.get() == 1:
            self.output_custom.config(state='disabled')
        else:
            self.output_custom.config(state='normal')
        if self.solution is not None: self.set_figure()

    def draw_figure(self, figure, loc=(0, 0)):
        """Draws a matplotlib figure onto the output canvas.
        Inspired by matplotlib source:
        https://matplotlib.org/3.1.1/gallery/user_interfaces/embedding_in_tk_sgskip.html

        Parameters
        ----------
        figure : matplotlib figure
            Figure to embed.
        loc : tuple(int, int)
            Location from top-left corner of the figure on the canvas.
        """
        # get canvas size
        cw, ch = self.output_canvas.winfo_width(), self.output_canvas.winfo_height()
        # check parameters for special resizing
        s = 1.75 if self.output_opt_3d.get() == 1 else 1.
        if self.output_opt_value.get() == 1:
            sw = min(cw//120, ch//60)
            figure.set_size_inches(s*sw, s*ch//120)
        else:
            figure.set_size_inches(s*ch//120, s*ch//120)
            
        # if necessary, destroy previous figure
        if self.fig_photo is not None:
            self.fig_photo.get_tk_widget().pack_forget()
        # create figure on canvas
        self.fig_photo = FigureCanvasTkAgg(figure, master=self.output_canvas)
        self.fig_photo.draw()
        self.fig_photo.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
    def set_figure(self, only_domain=False):
        """Sets the output figure of the application (either with or without a
        computed solution).
        
        Parameters
        ----------
        only_domain : bool, optional
            If true, the solution is ignored and the plain domain is plotted.
        """
        if self.discrete_domain is None: return
        
        plot_kwargs = { 'no_plot': True }
        plot_kwargs['value'] = int(self.output_opt_value.get())
        plot_kwargs['dim'] = 3 if self.output_opt_3d.get() == 1 else 2
        plot_kwargs['show_labels'] = int(self.output_opt_labels.get())
        plot_kwargs['show_triangulation'] = int(self.output_opt_triangulate.get())
        plot_kwargs['cmap'] = self.output_opt_cmap.get()
        
        old_fig = self.fig
        if only_domain:
            self.fig = self.discrete_domain.visualize(**plot_kwargs)
            self.draw_figure(self.fig)
        else:
            plotted_sol = self.solution
            if plotted_sol is not None and self.output_type.get() != 1:
                plotted_sol = self._parse_custom_str()
            self.fig = self.discrete_domain.visualize(z=plotted_sol, **plot_kwargs)
            self.draw_figure(self.fig)
        if old_fig is not None: plt.close(old_fig)
        
    def load_session(self):
        """Loads a SimpleFEMPy session (domain, variables, weak formulation and
        output settings)."""
        file_reader = filedialog.askopenfilename(title='Choose a SimpleFEMPy session file:')
        if len(file_reader) > 0:
            self.reset_session()
            
            regex = r'(?P<name>.*)\s*\$\s*(?P<value>.*)\n?'
            with open(file_reader, 'r') as f: content = f.read()
            matches = re.finditer(regex, content)
            for m in matches:
                n = m.group('name')
                v = m.group('value')
                if n == 'DOMAIN_NAME': self.discrete_domain_name.set(v)
                elif n == 'DOMAIN':
                    t = v.split('|')
                    if t[0] == '_Primitive_':
                        b = int(t[2])
                        self.domain_primitives_list.activate(SFEMPyApp.PRIMITIVES.index(t[1]))
                        self.set_primitive_params(t[1])
                        if t[1] == 'Line':
                            self.domain_param1.delete(0, tk.END)
                            self.domain_param1.insert(0, t[3])
                            self.domain_param2.delete(0, tk.END)
                            self.domain_param2.insert(0, t[4])
                            self.discrete_domain = DiscreteDomain.line(
                                float(t[3]), step=int(t[4]), nb_borders=b)
                        elif t[1] == 'Rectangle':
                            self.domain_param1.delete(0, tk.END)
                            self.domain_param1.insert(0, t[3])
                            self.domain_param2.delete(0, tk.END)
                            self.domain_param2.insert(0, t[4])
                            self.domain_param3.delete(0, tk.END)
                            self.domain_param3.insert(0, t[5])
                            self.discrete_domain = DiscreteDomain.rectangle(
                                float(t[3]), float(t[4]), step=int(t[5]),
                                nb_borders=b)
                        elif t[1] == 'Circle':
                            self.domain_param1.delete(0, tk.END)
                            self.domain_param1.insert(0, t[3])
                            self.domain_param2.delete(0, tk.END)
                            self.domain_param2.insert(0, t[4])
                            self.domain_param3.delete(0, tk.END)
                            self.domain_param3.insert(0, t[5])
                            self.discrete_domain = DiscreteDomain.circle(
                                float(t[3]), astep=int(t[4]), rstep=int(t[5]),
                                nb_borders=b)
                        elif t[1] == 'Ring':
                            self.domain_param1.delete(0, tk.END)
                            self.domain_param1.insert(0, t[3])
                            self.domain_param2.delete(0, tk.END)
                            self.domain_param2.insert(0, t[4])
                            self.domain_param3.delete(0, tk.END)
                            self.domain_param3.insert(0, t[5])
                            self.domain_param4.delete(0, tk.END)
                            self.domain_param4.insert(0, t[6])
                            self.discrete_domain = DiscreteDomain.ring(
                                float(t[3]), float(t[4]), astep=int(t[5]),
                                rstep=int(t[6]), nb_borders=b)
                        r = SFEMPyApp.PRIMITIVES.index(t[1])
                        self.domain_primitives_list.activate(r)
                        self.domain_primitives_list.selection_set(r)
                        self.domain_primitives_borders.delete(0, tk.END)
                        self.domain_primitives_borders.insert(0, t[2])
                        self.domain_reference = t[0]
                    else:
                        self.discrete_domain = DiscreteDomain.from_file(t[1])
                        self.domain_reference = t[1]
                elif n == 'VAR_FORMULATION': self.varf_str.set(v)
                elif n == 'VARIABLES':
                    vars = v.split('; ')
                    for v in vars:
                        self.add_variable()
                        self.vars[-1].set(v)
                elif n == 'OUTPUT_TYPE':
                    r = int(v)
                    self.output_type.set(r)
                    if r == 2: self.output_custom.config(state='normal')
                elif n == 'OUTPUT_CUSTOM':
                    if v == '_None_': pass
                    else: self.output_custom_str.set(v)
                elif n == 'OUTPUT_PARAMS':
                    params = v.split('|')
                    for p in params:
                        if 'cmap' in p:
                            cmap = p.split(':')[1]
                            self.output_opt_cmap.set(cmap)
                        elif p == 'value': self.output_opt_value.set(1)
                        elif p == '3D': self.output_opt_3d.set(1)
                        elif p == 'labels': self.output_opt_labels.set(1)
                        elif p == 'triangulate': self.output_opt_triangulate.set(1)
        self.set_figure()
        
    def save_session(self):
        """Saves a SimpleFEMPy session (domain, variables, weak formulation and
        output settings)."""
        file_reader = filedialog.asksaveasfilename(title='Save as...')
        if len(file_reader) > 0:
            with open(file_reader + '.sfpy', 'w') as f:
                f.write('DOMAIN_NAME$' + self.discrete_domain_name.get() + '\n')
                f.write('DOMAIN$')
                if self.domain_reference == '_Primitive_':
                    f.write('_Primitive_|' + self.domain_primitives_list.get(tk.ACTIVE))
                    f.write('|' + str(self.domain_primitives_borders.get()) + '|')
                    if self.domain_cur_primitive == 'Line':
                        f.write(self.domain_param1.get() + '|')
                        f.write(self.domain_param2.get())
                    elif self.domain_cur_primitive == 'Rectangle':
                        f.write(self.domain_param1.get() + '|')
                        f.write(self.domain_param2.get() + '|')
                        f.write(self.domain_param3.get())
                    elif self.domain_cur_primitive == 'Circle':
                        f.write(self.domain_param1.get() + '|')
                        f.write(self.domain_param2.get() + '|')
                        f.write(self.domain_param3.get())
                    elif self.domain_cur_primitive == 'Ring':
                        f.write(self.domain_param1.get() + '|')
                        f.write(self.domain_param2.get() + '|')
                        f.write(self.domain_param3.get() + '|')
                        f.write(self.domain_param4.get())
                else:
                    f.write('_File_|' + self.domain_reference)
                f.write('\n')
                f.write('VAR_FORMULATION$' + self.varf_str.get() + '\n')
                f.write('VARIABLES$' + '; '.join([v.get() for v in self.vars]) + '\n')
                f.write('OUTPUT_TYPE$' + str(self.output_type.get()) + '\n')
                if self.output_custom_str.get() != '':
                    f.write('OUTPUT_CUSTOM$' + self.output_custom_str.get() + '\n')
                else:
                    f.write('OUTPUT_CUSTOM$_None_\n')
                f.write('OUTPUT_PARAMS$')
                params = [ 'cmap:' + self.output_opt_cmap.get() ]
                if self.output_opt_value.get() == 1:
                    params.append('value')
                if self.output_opt_3d.get() == 1:
                    params.append('3D')
                if self.output_opt_labels.get() == 1:
                    params.append('labels')
                if self.output_opt_triangulate.get() == 1:
                    params.append('triangulate')
                f.write('|'.join(params))
                f.write('\n')
        
    def export_solution(self):
        """Exports a finite element solution to a VTK or CSV format."""
        if self.discrete_domain is None or self.solution is None: return
        file_reader = filedialog.asksaveasfilename(title='Save as...')
        if len(file_reader) > 0:
            ext = file_reader.split('.')[1].lower()
            if ext == 'csv':
                save_to_csv(file_reader, self.discrete_domain, self.solution)
            elif ext == 'vtu':
                save_to_vtk(file_reader, self.discrete_domain, self.solution)
            else:
                Logger.slog('Unknown type of export format: "{}".'.format(ext),
                            level='warning')

# create app instance
w, h = 1260, 768
app = SFEMPyApp(width=w, height=h)
TK_INSTANCE['app'] = app # (register app to settings object in case of errors)

# create custom buffer stream to report logging in the GUI screen
log_stream = SFEMPyStream(app)
Logger.sset_stream(log_stream)

# start main update
try:
    tk.mainloop()
except UnicodeDecodeError:
    pass
