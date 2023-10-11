import matplotlib.style as style
import math
from tools import *
from dos_fitfunc import *
from matplotlib.tri import Triangulation
from scipy.interpolate import griddata
from scipy.integrate import dblquad

from matplotlib.ticker import MaxNLocator


# Create a custom style sheet
custom_style = {
    'font.size': 16,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
}

# Register the custom style sheet
style.use(custom_style)

# Results will be saved in this folder.

dir_path = "$(DOS-Model)"

# Class for DOSModel - curve fitting and plotting.

plot_param()

class DOSModel:

    def __init__(self, xfile):
        self.xfile = xfile
        self.norm_df = None
        self.mod_df = None
        self.tdospred_df = None
        self.dospred_df = None
        self.pred_df = None
        
        self.fx = None
        self.popt = None
        self.pcov = None
        
        self.z_pred = None
        self.zpred_upper = None
        self.zpred_lower = None
        self.T_org = None
        self.t_org = None
        self.z_org = None
        
        self.T_mesh_org = None
        self.t_mesh_org = None
        self.z_mesh_org = None
        self.T_mesh_inv = None
        self.T_mesh_h = None 
        self.t_mesh_h = None
        self.z_mesh = None
        self.t_mesh_min = None

        self.deriv_T_somooth = None
        self.deriv__t_somooth = None

        self.deriv_T = None
        self.deriv__t = None
        

        # You need to save the normalize values somewhere, you never know!
        self.T_mesh = None
        self.t_mesh = None
        self.zpred_mesh = None

        self.xpath = None
        
        self.N = 500

        self.row_r = None

        self.fun_str = ''

        # Create result directory
        head, tail = ntpath.split(xfile)
        self.tail = tail.rsplit('.', 1)[0]
        resdir = self.tail
        xpath = os.path.join(dir_path, resdir)
        os.makedirs(xpath, exist_ok=True)
        self.xpath = xpath
    
    def save_plot(self, file_str):
        savefile = self.tail + file_str  + self.func_str + '.pdf'
        savefile = os.path.join(self.xpath, savefile)
        plt.savefig(savefile, bbox_inches='tight')
    
    def load_data(self):
        # Assign file, Read the CSV file into a DataFrame with column names 'T', 't', 'DOS'
        df = pd.read_csv(self.xfile, names=['T', 't', 'DOS'])

        columns = df.columns.tolist()

        # In csv file, Time is saved as ln(t), t is in hour. T as inverse of Temperature. Convert t to seconds. 
        t_new = 60 * 60 * np.exp(df['t'].values)

        # Create a new DataFrame with modified 'T' and 't' values and the original 'DOS' values
        self.mod_df = pd.DataFrame({'T': df['T'], 't': t_new, 'DOS': df['DOS']})
        
        # Save columns as numpy arrays, these are orignal values.
        self.T_org, self.t_org, self.z_org = self.mod_df['T'].to_numpy(), self.mod_df['t'].to_numpy(), self.mod_df['DOS'].to_numpy()

        # Initialize a MinMaxScaler object
        scaler = MinMaxScaler()

        # Perform normalization on the modified DataFrame
        normalized_data = scaler.fit_transform(self.mod_df)

        # Create a new DataFrame with the normalized data, using the same column names and index as the modified DataFrame
        self.norm_df = pd.DataFrame(normalized_data, columns=self.mod_df.columns, index=self.mod_df.index)

    def fit_curve(self, fx):
        self.fx = fx
        self.func_str = str(fx.__name__)
        
        # Fit the function 'fx' to the normalized data using curve_fit.
        # popt contains the optimized parameters, pcov contains the covariance matrix
        self.popt, self.pcov = curve_fit(self.fx, (self.norm_df['T'].values, self.norm_df['t'].values), self.norm_df['DOS'].values, maxfev=10000, method='lm')
        
        # Evaluate the fitted function 'fx' using the optimized parameters on the normalized data
        zpred = self.fx((self.norm_df['T'].values, self.norm_df['t'].values), *self.popt)

        # Store the normalized predictions in 'norm_pred'
        norm_pred = zpred

        # Find the minimum and maximum values of 'DOS' in the modified DataFrame
        z_min = np.min(self.mod_df['DOS'])
        z_max = np.max(self.mod_df['DOS'])

        # Rescale the normalized predictions to the original scale using the min-max scaling formula
        self.zpred_org = (norm_pred * (z_max - z_min)) + z_min

        self.row_r = CalFitIndex(self.z_org, self.zpred_org)

        return self.popt, self.pcov, self.row_r

    def do_fitting(self, fx):
        self.load_data()
        self.fit_curve(fx)
        self.smooth_df()
        self.construct_df()
        self.prepare_df()
        return None
    
    def smooth_df(self):
        
        T_vals = np.linspace(self.norm_df['T'].min(), self.norm_df['T'].max(), self.N)
        t_vals = np.linspace(self.norm_df['t'].min(), self.norm_df['t'].max(), self.N)
        
        # Create a meshgrid using the provided 'T_vals' and 't_vals'
        T_mesh, t_mesh = np.meshgrid(T_vals, t_vals)

        # Flatten the meshgrid coordinates and evaluate the fitted function 'fx'
        zpred_mesh = self.fx((T_mesh.flatten(), t_mesh.flatten()), *self.popt)

        # Reshape the flattened predictions to match the shape of the original meshgrid
        zpred_mesh = zpred_mesh.reshape(T_mesh.shape)
        
        self.T_mesh = T_mesh
        self.t_mesh = t_mesh
        self.zpred_mesh = zpred_mesh

        T_min = self.mod_df['T'].min()
        T_max = self.mod_df['T'].max()

        t_min = self.mod_df['t'].min()
        t_max = self.mod_df['t'].max()

        z_min = self.mod_df['DOS'].min()
        z_max = self.mod_df['DOS'].max()

        # Rescale the meshgrid coordinates using the provided minimum and maximum values
        T_mesh_org = T_mesh * (T_max - T_min) + T_min
        t_mesh_org = t_mesh * (t_max - t_min) + t_min

        
        z_min = self.mod_df['DOS'].min()
        z_max = self.mod_df['DOS'].max()
        
        zpred_mesh_org = zpred_mesh * (z_max - z_min) + z_min

        self.T_mesh_org = T_mesh_org
        self.t_mesh_org = t_mesh_org
        self.z_mesh_org = zpred_mesh_org

        self.t_mesh_h = self.t_mesh_org/3600
        self.t_mesh_min = self.t_mesh_org/60

        self.T_mesh_inv = (1 / T_mesh_org).astype(int)
        return None

    def plot_exp_pred(self):
        fig, ax = plt.subplots()
        set_axes(ax)

        plt.scatter(self.z_org, self.zpred_org, s = 30, c = 'black')

        # Draw an ideal predicted line
        ax.plot(self.z_org, self.z_org, '-', c = '#1E90FF', linewidth= 1.4)
        
        sigma = self.row_r[3]
        ax.errorbar(self.z_org, self.zpred_org, linewidth = 0.7, yerr=sigma, fmt='o', color='black', capsize=3,markersize = 0.7)
        if self.fx == 'R5-PrC.csv':
            ax.set_xlabel('Actual DOS, mpy')
            ax.set_ylabel('Predicted DOS, mpy')
        else:
            ax.set_xlabel('Actual DOS, %')
            ax.set_ylabel('Predicted DOS, %')
        fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)  # Adjust the values as needed
        self.save_plot('_pred_')
        plt.close(fig)
        return None

    def construct_df(self):
            # Create an empty list to store the dataframes
            df_list = []
            t_values = np.linspace(self.norm_df['t'].min(), self.norm_df['t'].max(), self.N)
            # Iterate over unique values of 'T'
            for T in self.norm_df['T'].unique():
                # Generate 't' values with N data points
                # Create a dataframe with 't' values and the corresponding 'DOS' values for the current 'T'
                df_temp = pd.DataFrame({'t': t_values, 'DOS': self.fx((T, t_values), *self.popt)})

                # Append the dataframe to the list
                df_list.append(df_temp)

            # Concatenate the dataframes from the list into a single dataframe
            tdos_df = pd.concat(df_list, keys=self.norm_df['T'].unique(), names=['T'])

            # Reset the index of the tdos_df dataframe
            tdos_df = tdos_df.reset_index(level='T')

            # Create a new dataframe for reconstructed values
            self.tdospred_df = pd.DataFrame()

            # Reconvert the 'T' column
            self.tdospred_df['T'] = (tdos_df['T'] * (self.mod_df['T'].max() - self.mod_df['T'].min())) + self.mod_df['T'].min()

            # Reconvert the 't' column
            self.tdospred_df['t'] = ((tdos_df['t'] * (self.mod_df['t'].max() - self.mod_df['t'].min())) + self.mod_df['t'].min())/60

            # Reconvert the 'DOS' column
            self.tdospred_df['DOS'] = (tdos_df['DOS'] * (self.mod_df['DOS'].max() - self.mod_df['DOS'].min())) + self.mod_df['DOS'].min()
            return None

    def plot_2d(self):
        
        fig, ax = plt.subplots()

        num_unique_T = len(self.mod_df['T'].unique())

        # Reverse the order of T values
        unique_T = self.tdospred_df['T'].unique()[::-1]

        # Iterate over unique values of T in mod_df['T']
        for i, T in enumerate(unique_T):
            # Filter the dataframe for the current T value
            df_temp = self.mod_df[self.mod_df['T'] == T]
            # Plot t vs. DOS as a scatter plot with the corresponding color
            ax.scatter(df_temp['t'], df_temp['DOS'], color=xcolors[i])

        # Iterate over unique values of T in tdospred_df['T']
        for i, T in enumerate(unique_T):
            # Filter the dataframe for the current T value
            tdospred_temp = self.tdospred_df[self.tdospred_df['T'] == T]
            # Plot t vs. DOS as a solid line with the corresponding color
            ax.plot(60*tdospred_temp['t'], tdospred_temp['DOS'], label=f'T = {int(1/T)} K', color=xcolors[i], linestyle='-', linewidth = 1.5)

        ax.set_xlabel(r'$t$, s')
        ax.set_ylabel('DOS, %')
        ax.legend()
        set_axes(ax, xlog = True)
        fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)  # Adjust the values as needed
        self.save_plot('_tdos_')
        plt.close(fig)

        return None

    def prepare_df(self):
        # Create an empty list to store the dataframes
        df_list = []

        # Iterate over unique values of 'T'
        for T in self.norm_df['T'].unique():
            # Compute the minimum and maximum 't' values for the current 'T'
            t_min = self.norm_df[self.norm_df['T'] == T]['t'].min()
            t_max = self.norm_df[self.norm_df['T'] == T]['t'].max()

            # Generate 't' values with N data points within the range of t_min and t_max
            t_values = np.linspace(t_min, t_max, self.N)

            # Create a dataframe with 't' values and the corresponding 'DOS' values for the current 'T'
            df_temp = pd.DataFrame({'t': t_values, 'DOS': self.fx((T, t_values), *self.popt)})

            # Rescale 't' and 'DOS' values
            df_temp['t'] = (df_temp['t'] * (self.mod_df['t'].max() - self.mod_df['t'].min())) + self.mod_df['t'].min()
            df_temp['DOS'] = (df_temp['DOS'] * (self.mod_df['DOS'].max() - self.mod_df['DOS'].min())) + self.mod_df['DOS'].min()

            # Append the dataframe to the list
            df_list.append(df_temp)

        # Concatenate the dataframes from the list into a single dataframe
        tdos_df = pd.concat(df_list, keys=self.norm_df['T'].unique(), names=['T'])

        # Reset the index of the tdos_df dataframe
        tdos_df = tdos_df.reset_index(level='T')

        # Create a new dataframe for reconstructed values
        self.dospred_df = pd.DataFrame()

        # Reconvert the 'T' column
        self.dospred_df['T'] = (tdos_df['T'] * (self.mod_df['T'].max() - self.mod_df['T'].min())) + self.mod_df['T'].min()

        # Copy the 't' and 'DOS' columns from tdos_df to self.dospred_df
        self.dospred_df['t'] = tdos_df['t']
        self.dospred_df['DOS'] = tdos_df['DOS']
        return None

    def plot_2d_local(self):
            # Create a color map based on the number of unique T values
            fig, ax = plt.subplots()
            num_unique_T = len(self.mod_df['T'].unique())
            cmap = plt.get_cmap('tab10')
            colors = cmap(np.linspace(0, 1, num_unique_T))
            # Reverse the order of T values
            unique_T = self.dospred_df['T'].unique()[::-1]
            # Iterate over unique values of T in mod_df['T']
            for i, T in enumerate(unique_T):
                # Filter the dataframe for the current T value
                df_temp = self.mod_df[self.mod_df['T'] == T]
                # Plot t vs. DOS as a scatter plot with the corresponding color
                ax.scatter(df_temp['t'], df_temp['DOS'], color=xcolors[i])
            # Iterate over unique values of T in tdospred_df['T']
            for i, T in enumerate(unique_T):
                # Filter the dataframe for the current T value
                tdospred_temp = self.dospred_df[self.dospred_df['T'] == T]
                # Plot t vs. DOS as a solid line with the corresponding color
                ax.plot(tdospred_temp['t'], tdospred_temp['DOS'], label=f'T = {int(1/T)} K', color=xcolors[i], linestyle='-', linewidth = 1.5)

            ax.set_xlabel(r'$t$, s')
            ax.set_ylabel('DOS, %')
            ax.legend()
            set_axes(ax, xlog = True)
            fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)  # Adjust the values as needed
            self.save_plot('_dos_')
            plt.close(fig)
            return None

    def plot_3d(self):
        # Create a 3D plot with the rescaled meshgrid coordinates and predicted values
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        #set_matplotlib_fonts(ax)
        surf = ax.plot_surface(self.T_mesh_inv, self.t_mesh_h, self.z_mesh_org, cstride = 5, rstride = 5, cmap='coolwarm_r', alpha = 0.75)
        
        ax.w_xaxis.set_pane_color((0.4, 0.4, 0.4, 0.1))
        ax.w_yaxis.set_pane_color((0.4, 0.4, 0.4, 0.1))
        ax.w_zaxis.set_pane_color((0.4, 0.4, 0.4, 0.1))

        set_axes(ax, xlog = False)
        
        cbar = fig.colorbar(surf, shrink = 0.9, pad = 0.08)
        tick_font_size = 12.5
        cbar.ax.tick_params(labelsize=tick_font_size)

        cset = ax.contourf(self.T_mesh_inv, self.t_mesh_h, self.z_mesh_org, cmap='coolwarm_r', zdir='z', offset= 1.12*np.max(self.z_mesh_org), alpha = 0.6, levels = 20)
        cset1 = ax.contour(self.T_mesh_inv, self.t_mesh_h, self.z_mesh_org, cmap='coolwarm_r', zdir='y', offset= -1.15*np.min(self.t_mesh_h), alpha = 0.9, linewidths = 1)
        cset2 = ax.contour(self.T_mesh_inv, self.t_mesh_h, self.z_mesh_org, cmap='coolwarm_r', zdir='x', offset= 1.02*np.max(self.T_mesh_inv), alpha = 1.0, linewidths = 1.2)
        
        
        ax.grid(False)
     
        ax.set_xlabel(r'$T$, K', labelpad = 4)
        ax.set_ylabel(r'$t$, h', labelpad = 5)
        ax.set_zlabel('DOS, %')
        ax.tick_params(axis='x', pad= 1)
        ax.tick_params(axis='y', pad = 2)
        ax.tick_params(axis='z', pad = 1)

        # Set the number of major ticks for all three axes
        num_ticks = 5  # Change this to the desired number of ticks

        ax.xaxis.set_major_locator(MaxNLocator(num_ticks))
        ax.yaxis.set_major_locator(MaxNLocator(num_ticks))
        ax.zaxis.set_major_locator(MaxNLocator(num_ticks))
       
        ax.zaxis.set_minor_locator(AutoMinorLocator())
            
        fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)  # Adjust the values as needed
        self.save_plot('_3d_')
        plt.close(fig)

        return None

    def plot_contour(self):
        fig, ax = plt.subplots()
        
        dos_levels = 12
        #color_map = contour_color_map()
        filled_contour_plot = ax.tricontourf(self.t_mesh_org.flatten(), self.T_mesh_inv.flatten(), self.z_mesh_org.flatten(), cmap = 'coolwarm_r', levels = dos_levels)
        # contour_plot = ax.tricontour(self.t_mesh_org.flatten(), self.T_mesh_inv.flatten(), self.z_mesh_org.flatten(), colors = 'white', linewidths= 0.1, levels = dos_levels)
        # ax.clabel(contour_plot, contour_plot.levels, inline=True, fmt='%1.1f', fontsize = 15)


        cbar = fig.colorbar(filled_contour_plot, ax=ax)
        if self.xfile == 'R5-PrC.csv':
            cbar.set_label('DOS, mpy')    
        else:
            cbar.set_label('DOS, %')
        cbar.ax.tick_params(labelsize=14)

        ax.set_xlabel(r'$t$, s')
        ax.set_ylabel(r'$T$, K')
        set_axes(ax, xlog = False)
        fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)  # Adjust the values as needed
        self.save_plot('_no_log_con_')
        set_axes(ax, xlog = True)    
        fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)  # Adjust the values as needed
        self.save_plot('_con_')
        plt.close(fig)

        fig, ax = plt.subplots()
        
        dos_levels = [10, 20, 30, 40]

        contour_plot = ax.tricontour(self.t_mesh_org.flatten(), self.T_mesh_inv.flatten(), self.z_mesh_org.flatten(), \
                                     colors = 'k', linewidths= 1.25, levels = dos_levels)
        ax.clabel(contour_plot, contour_plot.levels, inline=True, fmt='%1.1f', fontsize = 15)

        xcontour_plot = ax.tricontour(self.t_org, 1/self.T_org, self.z_org, colors = '#F00B5F', linestyles = '--', \
                                      linewidths = 1.5, levels = dos_levels)
        # ax.clabel(xcontour_plot, xcontour_plot.levels, inline=True, fmt='%1.1f', fontsize = 15)
               
        ax.set_xlabel(r'$t$, s')
        ax.set_ylabel(r'$T$, K')
        set_axes(ax, xlog = False)    
        fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)  # Adjust the values as needed
        self.save_plot('_xtts_')
        set_axes(ax, xlog = True)    
        fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)  # Adjust the values as needed
        self.save_plot('_tts_')
        plt.close(fig)

    def one_deriv(self, fx):
        fig, ax = plt.subplots()
        X = (self.T_mesh.flatten(), self.t_mesh.flatten())
        deriv_T = fx(X, *(self.popt))
        deriv_T_shaped = deriv_T.reshape(self.T_mesh.shape)
        
        if fx == dT_F0:
            self.deriv_T = deriv_T_shaped
        elif fx == dt__F0:
            self.deriv__t = deriv_T_shaped
        elif fx == dT_F1:
            self.deriv_T = deriv_T_shaped
        elif fx == dt__F1:
            self.deriv__t = deriv_T_shaped
        else:
            print("Error.")
            return -1

        # print(deriv_T_shaped)
        num_levels = 10
        min_value = min(deriv_T)
        max_value = max(deriv_T)
        dos_levels = np.linspace(min_value, max_value, 10)
        dos_levels = np.insert(dos_levels, np.searchsorted(dos_levels, 0), 0)
        contour_plot = ax.contour(self.t_mesh, self.T_mesh,  deriv_T_shaped, levels=dos_levels, colors='black', linewidths = 0.7)
        ax.clabel(contour_plot, contour_plot.levels, inline=True, fmt='%1.1f', fontsize = 16)
        ax.set_xlabel(r'$y$')
        ax.set_ylabel(r'$x$')
        set_axes(ax, xlog = False)    
        # plt.tight_layout()
        fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)  # Adjust the values as needed
        savefile = '_' + str(fx.__name__) + '-'
        self.save_plot(savefile)
        plt.close(fig)
        return None

    def one_deriv_smooth(self, fx):
        fig, ax = plt.subplots()
        X = (self.T_mesh.flatten(), self.t_mesh.flatten())
        deriv_T = fx(X, *(self.popt))
        deriv_T_shaped = deriv_T.reshape(self.T_mesh.shape)

        if fx == dT_F0:
            self.deriv_T_smooth = deriv_T_shaped
        elif fx == dt__F0:
            self.deriv__t_smooth = deriv_T_shaped
        elif fx == dT_F1:
            self.deriv_T_smooth = deriv_T_shaped
        elif fx == dt__F1:
            self.deriv__t_smooth = deriv_T_shaped
        else:
            print("Error.")
            return -1

        dos_levels = 13
        filled_contour_plot = ax.contourf(self.t_mesh, self.T_mesh, deriv_T_shaped, cmap = 'coolwarm_r', levels = dos_levels)
        cbar = fig.colorbar(filled_contour_plot, ax=ax)
        contour_plot = ax.contour(self.t_mesh, self.T_mesh,  deriv_T_shaped, levels=dos_levels, colors='white', linewidths = 1.24)
        # ax.clabel(contour_plot, contour_plot.levels, inline=True, fmt='%1.1f', fontsize = 16)
        ax.set_xlabel(r'$y$')
        ax.set_ylabel(r'$x$')
        set_axes(ax, xlog = False)    
        # plt.tight_layout()
        fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)  # Adjust the values as needed
        savefile = '_' + str(fx.__name__) + '-smooth-'
        self.save_plot(savefile)
        plt.close(fig)
        return None
    
    def calc_deriv(self):
        if self.fx == F0:
            self.one_deriv(dT_F0)
            self.one_deriv(dt__F0)
        elif self.fx == F1:
            self.one_deriv(dT_F1)
            self.one_deriv(dt__F1)
        else:
            print("Derivative not found.")
            return None
        return None
    
    def calc_deriv_smooth(self):
        if self.fx == F0:
            self.one_deriv_smooth(dT_F0)
            self.one_deriv_smooth(dt__F0)
        elif self.fx == F1:
            self.one_deriv_smooth(dT_F1)
            self.one_deriv_smooth(dt__F1)
        else:
            print("Derivative not found.")
            return None
        return None

    def calculate_ranges(self, z0):
        T_range = []
        t_range = []
        # First, dos_level = z0 should be normalized
        z0_norm = (z0 - min(self.z_org))/(max(self.z_org) - min(self.z_org))
        
        for T, t in zip(self.T_mesh.flatten(), self.t_mesh.flatten()):
            z = self.fx((T, t), *self.popt)
            if np.isclose(z, z0_norm, rtol=1e-4, atol=1e-4):
                T_range.append(T)
                t_range.append(t)
        
        if not T_range or not t_range:
            return None
        
        T_min_z0 = min(T_range) if T_range else None
        T_max_z0 = max(T_range) if T_range else None
        t_min_z0 = min(t_range) if t_range else None
        t_max_z0 = max(t_range) if t_range else None

        
        
        T_min_z0 = min(T_range)*(max(self.T_org) - min(self.T_org)) + min(self.T_org)
        T_max_z0 = max(T_range)*(max(self.T_org) - min(self.T_org)) + min(self.T_org)
        
        t_min_z0 = min(t_range)*(max(self.t_org) - min(self.t_org)) + min(self.t_org)
        t_max_z0 = max(t_range)*(max(self.t_org) - min(self.t_org)) + min(self.t_org)

        T_min_z0 = 1/T_min_z0
        T_max_z0 = 1/T_max_z0
        t_min_z0 = int(t_min_z0/3600)
        t_max_z0 = int(t_max_z0/3600)
        
        return int(T_max_z0), int(T_min_z0),  round(t_min_z0, 2), round(t_max_z0, 2)

    def integrand(self, T, t, z0):
        return self.fx((T, t), *self.popt) - z0

    def integrate(self, num_levels = 5):
        dos_levels = np.linspace(min(self.z_org), max(self.z_org), num_levels)
        dos_levels = [round(element, 1) for element in dos_levels]
        results = []
        for z0 in dos_levels:
            range_values = self.calculate_ranges(z0)
            if range_values is None:
                continue
            T_min_z0, T_max_z0, t_min_z0, t_max_z0 = range_values
            # result, _ = dblquad(self.integrand, T_min_z0, T_max_z0, lambda t: t_min_z0, lambda t: t_max_z0, args=(z0))
            results.append([z0, T_min_z0, T_max_z0, t_min_z0, t_max_z0])
        # Create pandas DataFrame
        df = pd.DataFrame(results, columns=['z0', 'T_min_z0', 'T_max_z0', 't_min_z0', 't_max_z0'])

        # Save DataFrame to Excel file
        savefile = self.tail + '_dos_range_'  + self.func_str + '.xlsx'
        savefile = os.path.join(self.xpath, savefile)
        df.to_excel(savefile, index=False)

        # # Generate LaTeX code from DataFrame
        # latex_code = df.to_latex(index=False)
        # savefile = self.tail + '_latex_dos_range_'  + self.func_str + '.tex'
        # savefile = os.path.join(self.xpath, savefile)
        # # Save LaTeX code to tex file
        # with open(savefile, 'w') as file:
        #     file.write(latex_code)
        return None

    def calc_range(self):
        dos_levels = [9]
        results = []
        xstr = self.func_str
        for z0 in dos_levels:
            range_values = self.calculate_ranges(z0)
            if range_values is None:
                continue
            T_min_z0, T_max_z0, t_min_z0, t_max_z0 = range_values
            xstr += str(z0)
            # result, _ = dblquad(self.integrand, T_min_z0, T_max_z0, lambda t: t_min_z0, lambda t: t_max_z0, args=(z0))
            results.append([z0, T_min_z0, T_max_z0, t_min_z0, t_max_z0])
        # Create pandas DataFrame
        df = pd.DataFrame(results, columns=['z0', 'T_min_z0', 'T_max_z0', 't_min_z0', 't_max_z0'])

        # Save DataFrame to Excel file
        savefile = self.tail + '_range_'  + xstr + '.xlsx'
        savefile = os.path.join(self.xpath, savefile)
        df.to_excel(savefile, index=False)

        # # Generate LaTeX code from DataFrame
        # latex_code = df.to_latex(index=False)
        # savefile = self.tail + '_latex_dos_range_'  + self.func_str + '.tex'
        # savefile = os.path.join(self.xpath, savefile)
        # # Save LaTeX code to tex file
        # with open(savefile, 'w') as file:
        #     file.write(latex_code)
        
        return None

    def plot_contour_norm(self):
        fig, ax = plt.subplots()

        dos_levels = 5
        color_map = contour_color_map()
        filled_contour_plot = ax.tricontourf(self.t_mesh.flatten(), self.T_mesh.flatten(), self.zpred_mesh.flatten(), cmap = 'coolwarm_r', levels = dos_levels)
        # contour_plot = ax.tricontour(self.t_mesh.flatten(), self.T_mesh.flatten(), self.zpred_mesh.flatten(), colors = 'k', linewidths= 0.9, levels = dos_levels)
        # ax.clabel(contour_plot, contour_plot.levels, inline=True, fmt='%1.1f', fontsize = 16)


        cbar = fig.colorbar(filled_contour_plot, ax=ax)
        cbar.set_label('DOS, %')
        cbar.ax.tick_params(labelsize=14)

        ax.set_xlabel(r'$y$')
        ax.set_ylabel(r'$x$')
        
        set_axes(ax, xlog = False)
        fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)  # Adjust the values as needed
        self.save_plot('_cont_n_')
        plt.close(fig)

    def plot_contour_norm_local(self):
        fig, ax = plt.subplots()

        # Initialize a MinMaxScaler object
        scaler = MinMaxScaler()

        # Perform normalization on the modified DataFrame
        normalized_data = scaler.fit_transform(self.tdospred_df)

        # Create a new DataFrame with the normalized data, using the same column names and index as the modified DataFrame
        tdos_temp = pd.DataFrame(normalized_data, columns=self.tdospred_df.columns, index=self.tdospred_df.index)
        
        tvals = tdos_temp['t'].to_numpy()
        Tvals = tdos_temp['T'].to_numpy()
        dosvals = tdos_temp['DOS'].to_numpy()

        dos_levels = 5

        contour_plot = ax.tricontour(tvals, Tvals, dosvals, colors = 'k', linewidths= 0.9, levels = dos_levels)
        ax.clabel(contour_plot, contour_plot.levels, inline=True, fmt='%1.1f', fontsize = 16)
        ax.set_xlabel(r'$y$, s')
        ax.set_ylabel(r'$x$, K$^{-1}$')
        set_axes(ax, xlog = False)
        fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)  # Adjust the values as needed
        self.save_plot('_cont_n_local_')
        plt.close(fig)
        return fig
    

    
