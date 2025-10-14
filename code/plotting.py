# -*- coding: utf-8 -*-
import glob, os, shutil, json, scipy, math
import nibabel as nib
import numpy as np
import pingouin as pg
import pandas as pd

# Time computation libraries
from joblib import Parallel, delayed
import time
from tqdm import tqdm

# Nilearn library
from nilearn.maskers import NiftiMasker
from nilearn import image
from nilearn import plotting
from nilearn import surface

# Sklearn library
from sklearn.feature_selection import mutual_info_regression
from sklearn import decomposition

# Statistics library
from scipy import stats
from scipy.ndimage import center_of_mass

from statsmodels.stats.multitest import multipletests

# plotting libraries
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import seaborn as sns


class Plotting:
    '''
    The Seed2voxels class is used to run correlation analysis
    Attributes
    ----------
    config : dict
    
    '''
    
    def __init__(self, config_file, outputdir):
        with open(config_file) as config_f:
            self.config = json.load(config_f) # load config info
        self.config_file=config_file

        if outputdir:
            self.outputdir=outputdir
        else:
            self.outputdir= self.config["project_dir"] +self.config["analysis_dir"]

    
        #>>> create output directory if needed -------------------------------------
        #print(self.config['main_dir'] + self.config["analysis_dir"])
        if not os.path.exists(self.outputdir):
            os.mkdir(self.outputdir)

        
    def plot_heatmap(self,matrix=None, labels=None, networks=None,networks_y=None, metric='corr', index_raw=None,index_col=None,
                        edgecolor='black',half=None,
                        ax=None, figsize=(6.4, 4.8), xlabels=None, ylabels=None,
                        cmap='viridis',vmin=-1,vmax=1,
                        xlabelrotation=90, ylabelrotation=0, cbar=True,
                        square=True, xticklabels=None, yticklabels=None,
                        mask_diagonal=True,  output_f="",save= False,**kwargs):
        """
        Plot 'matrix' as heatmap

        Parameters
        ----------
        Matrix : (n,n) array_like or df
            Correlation matrix
        labels : (n,1) array_like
            Label assignments for each raw and col of the matrix
        networks : (n,1) array_like
            netowrks assignments for sub labels of the matrix
        metric: string
            if 'matrix' is a dataframe you should precise the variable column name default: 'corr'
        
        index_raw
            default: None
        
        index_col
            default: None

        edgecolor : str, optional
            Color for lines demarcating networks boundaries. Default: 'black'
        ax : matplotlib.axes.Axes, optional
            Axis on which to plot the heatmap. If none provided, a new figure and
            axis will be created. Default: None
        figsize : tuple, optional
            Size of figure to create if `ax` is not provided. Default: (20, 20)
        {x,y}labels : list, optional
            List of labels on {x,y}-axis for each network in `networks`. The
            number of labels should match the number of unique networks.
            Default: None
        {x,y}labelrotation : float, optional
            Angle of the rotation of the labels. Available only if `{x,y}labels`
            provided. Default : xlabelrotation: 90, ylabelrotation: 0
        square : bool, optional
            Setting the matrix with equal aspect. Default: True
        {x,y}ticklabels : list, optional
            Incompatible with `{x,y}labels`. List of labels for each entry (not
            community) in `data`. Default: None
        cbar : bool, optional
            Whether to plot colorbar. Default: True
        mask_diagonal : bool, optional
            Whether to mask the diagonal in the plotted heatmap. Default: True
        kwargs : key-value mapping
            Keyword arguments for `plt.pcolormesh()`

        Returns
        -------
        ax : matplotlib.axes.Axes
            Axis object containing plot
        """
        
        if matrix is None:
            raise ValueError('Please provide a (n,n) array_like or df as input for "matrix" ')

        # transform a dataframe into matrix related to raw and columns names and metric values
        if isinstance(matrix, pd.DataFrame):
            matrix = matrix.pivot_table(index=index_raw, columns=index_col, values=metric,sort=False) # sort=False  > To preserve the original order of the index values, otherwise by default it is alphabetic
            
            if labels:
                matrix=matrix.reindex(index=labels, columns=labels)


        for t, label in zip([xticklabels, yticklabels], [xlabels, ylabels]):
            if t is not None and label is not None:
                raise ValueError('Cannot set both {x,y}labels and {x,y}ticklabels')


        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        inds_x = list(range(matrix.shape[0])) # this part could be update to plot only subpart of the matrix
        inds_y = list(range(matrix.shape[1])) # this part could be update to plot only subpart of the matrix
        

        # mask the diagonal
        mask = np.zeros_like(matrix, dtype=bool)

        if mask_diagonal:
            np.fill_diagonal(mask, True)

        if half == 'upper':
            mask |= np.tril(np.ones_like(mask, dtype=bool), k=-1)  # mask lower triangle
        elif half == 'lower':
            mask |= np.triu(np.ones_like(mask, dtype=bool), k=1)   # mask upper triangle

        # Apply the mask
        plot_matrix = np.ma.masked_where(mask, matrix)


        #plot the matrix
        coll = ax.pcolormesh(plot_matrix, edgecolor='none', cmap=cmap,vmin=vmin, vmax=vmax)
        ax.set(xlim=(0, plot_matrix.shape[1]), ylim=(0, plot_matrix.shape[0]))

        # set equal aspect
        if square:
            ax.set_aspect('equal')

        for side in ['top', 'right', 'left', 'bottom']:
            ax.spines[side].set_visible(False)

        # invert the y-axis so it looks "as expected"
        ax.invert_yaxis()

        # plot the colorbar
        if cbar:
            cb = ax.figure.colorbar(coll)
            if kwargs.get('rasterized', False):
                cb.solids.set_rasterized(True)

        # draw borders around networks
        # Default networks_y to networks_x if not specified
        if networks is not None:
            if networks_y is None:
                networks_y = networks
                networks_x = networks
            else:
                networks_x =networks

            # Loop through each unique network in 'networks', with 'i' as a 1-based index
            unique_networks = []
            bounds = []
            if np.array_equal(networks_x, networks_y):  # Symmetric case
                for network in networks:
                    if network not in unique_networks:
                        unique_networks.append(network)

                unique_networks_x=unique_networks; unique_networks_y=unique_networks
                for i, network in enumerate(unique_networks):
                    i=+1
                    
                    ind = np.where(np.array(networks) == network)

                    if len(ind) > 0:
                        bounds.append(np.min(ind))
                bounds.append(len(networks))
                bounds_x=bounds
                bounds_y=bounds

            
            else:  # Non-symmetric case
                # Compute bounds for row networks (y-axis)
                unique_networks_y = list(dict.fromkeys(networks_y))
                bounds_y = []
                for network in unique_networks_y:
                    ind = np.where(np.array(networks_y) == network)[0]
                    if len(ind) > 0:
                        bounds_y.append(np.min(ind))
                bounds_y.append(len(networks_y))

                # Compute bounds for column networks (x-axis)
                unique_networks_x = list(dict.fromkeys(networks_x))
                bounds_x = []
                

                for network in unique_networks_x:
                    ind = np.where(np.array(networks_x) == network)[0]
                    if len(ind) > 0:
                        bounds_x.append(np.min(ind))
                bounds_x.append(len(networks_x))

            initial_bottom, initial_top = ax.get_ylim()
            initial_left, initial_right = ax.get_xlim()

            
            gap_fraction = 0.05  # Gap between network segments as a fraction of the network's size
    
            for start, end in zip(bounds_x[:-1], bounds_x[1:]):
                network_size = end - start
                gap = network_size * gap_fraction
                ax.hlines(y=initial_bottom+5 , xmin=start + gap, xmax=end - gap, colors=edgecolor, linestyles='-', linewidth=5)
                
            for start, end in zip(bounds_y[:-1], bounds_y[1:]):
                network_size = end - start
                gap = network_size * gap_fraction
                ax.vlines(x=-5, ymin=start + gap, ymax=end - gap, colors=edgecolor, linestyles='-', linewidth=4)
                

            ax.set_xlim(left=-5)
            ax.set_ylim(bottom=initial_bottom+5,top=initial_top)


            if xlabels is not None or ylabels is not None:
                if xlabels is not None:
                    tickloc_x = []
                    for loc in range(len(bounds_x) - 1):
                        tickloc_x.append(np.mean((bounds_x[loc], bounds_x[loc + 1])))
                    

                    if len(tickloc_x) != len(unique_networks_x):
                        raise ValueError('Number of labels do not match the number of '
                                            'unique communities.')
                    else:
                        ax.set_xticks(tickloc_x)
                        ax.set_xticklabels(labels=unique_networks_x, rotation=xlabelrotation,fontsize=9)
                        ax.tick_params(left=False, bottom=False)

                if ylabels is not None:
                    tickloc_y=[]
                    for loc in range(len(bounds_y) - 1):
                        tickloc_y.append(np.mean((bounds_y[loc], bounds_y[loc + 1])))
                    # make sure number of labels match the number of ticks
                    if len(tickloc_y) != len(unique_networks_y):
                        raise ValueError('Number of labels do not match the number of '
                                            'unique communities.')
                    else:
                        ax.set_yticks(tickloc_y)
                        ax.set_yticklabels(labels=unique_networks_y, rotation=ylabelrotation,fontsize=9)
                        ax.tick_params(left=False, bottom=False)

        if xticklabels is not None:
            labels_ind = [xticklabels[i] for i in inds_x]
            ax.set_xticks(np.arange(len(labels_ind)) + 0.5)
            ax.set_xticklabels(labels_ind, rotation=90)
        if yticklabels is not None:
            labels_ind = [yticklabels[i] for i in inds_y]
            ax.set_yticks(np.arange(len(labels_ind)) + 0.5)
            ax.set_yticklabels(labels_ind)
        
        plt.show()
        if save==True:
            if output_f=="":
                output_f="corr_matrix.pdf"
            
            ax.figure.savefig(output_f, dpi=300,format='pdf', bbox_inches='tight')

            plt.close()  # Close the figure to release memory


        return ax
    
    def matrix(self,matrix=None,labels=None,metric='corr',input_type="dataframe",cmap_corr='seismic',heatmap_pval=None,output_dir=None,output_tag='',index_raw=None,index_col=None,raw=None,columns=None,vmin=-0.5, vmax=0.5,indiv=False,group=False,save=False):
        '''
        Create matrix of correlation
        cmap:
            colormap
        files: [filemame]
            Filename that contain group or individuals values like a dataframe
        input_type:
            The input could be an array with same number of raw and columns or a dataframe with one columns for each corr values

        output_dir: 
        redo: bool
            put redo=true to rerun the analyse

        '''
        # read input file
        if indiv==True:
            df_indiv=matrix
            #Plots
            ncols = 4 # Number of columns
            nrows = math.ceil(len(self.config["participants_IDs"]) / ncols) # Calculate the number of rows needed (rounding up)
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(23, nrows*5))  # Adjust figsize as needed
            axes = axes.flatten()
            for ID_nb, ID_name in enumerate(self.config["participants_IDs"]):
                ax=axes[ID_nb]
                self._plot_matrices(heatmap_data=df_indiv[ID_nb],mask=None,raw=raw,columns=columns,labels=None,index_raw=None,index_col=None,center=0,cmap=cmap_corr,norm=None,cbar=False,vmin=vmin,vmax=vmax,ax=ax)
                ax.set_title('Correlation matrix sub-' + ID_name)# Set the title for each subplot

            # Hide any empty subplots (if participants < total subplots)
            for ax in axes[len(self.config["participants_IDs"]):]:
                ax.axis('off')
            
            output_dir=self.firstlevel_dir + self.config["extract_corr"]["output_dir"]
            output_fig=output_dir + "meancorr_matrix_" + output_tag 

        if group:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 6))  # Adjust figsize as needed
            # Pivot data for correlation and p-values
            if input_type=="dataframe":
                heatmap_corr = matrix.pivot_table(index=raw, columns=columns, values=metric)
            else:
                heatmap_corr=matrix
            #np.fill_diagonal(heatmap_corr.values, 0)

            if heatmap_pval is not None:
                #data_pval = df.pivot_table(index=index_raw, columns=columns, values='p_value')  # Assuming you have p-value column
                #heatmap_pval.fillna(1, inplace=True)
                mask_corr = np.triu(np.ones_like(heatmap_corr, dtype=bool)) # Mask the upper triangle
                mask_pval = np.tril(np.ones_like(heatmap_pval, dtype=bool), k=-1) # Mask the lower triangle
                mask_pval_tria = np.tril(np.ones_like(heatmap_pval, dtype=bool), k=-1) # Mask the lower triangle
                mask_pval_thr = heatmap_pval < 0.05
                combined_mask = mask_pval_tria| ~mask_pval_thr  # Mask the upper triangle and values where p >= 0.05

                #create a cmap for pval
                new_matrix = np.zeros_like(heatmap_pval, dtype=int)
                new_matrix[heatmap_pval < 0.001] = 3
                new_matrix[(heatmap_pval >= 0.001) & (heatmap_pval < 0.01)] = 2
                new_matrix[(heatmap_pval >= 0.01) & (heatmap_pval < 0.05)] = 1
                # Leave values > 0.05 as 0
                masked_new_matrix = np.where(combined_mask, np.nan, new_matrix)  # Use np.nan for masked areas  

                # Define your custom colormap
                colors = ['#e5e3e3',  # 0 (white)
                        '#dedada',  # 1 (gold)
                        '#bcbaba',  # 2 (orange)
                        '#a4a3a3']  # 3 (red)
        
                discretized_colormap = ListedColormap(colors)

                has_one_value = np.any(masked_new_matrix == 1)

                if has_one_value:
                    print("There is at least one value of 1 in masked_new_matrix.")
                else:
                    print("There are no values of 1 in masked_new_matrix.")

                self._plot_matrices(heatmap_data=masked_new_matrix,mask=None,raw=raw,columns=columns,center=1.5,labels=labels,
                                    index_raw=None,index_col=None,cmap=discretized_colormap,norm=None,cbar=False,
                                    vmin=0,vmax=3,ax=ax)
    
            else:
                mask_corr=None

            
            self._plot_matrices(heatmap_data=heatmap_corr,mask=mask_corr,raw=raw,columns=columns,labels=labels,
                                    index_raw=index_raw,index_col=index_col,cmap=cmap_corr,norm=None,cbar=True,center=0,
                                    vmin=vmin,vmax=vmax,ax=ax)
            
            
            ax.set_title('Correlation matrix ' + output_tag)  # Set the title for the plot
                
            # Output filename
            output_fig = output_dir + "meancorr_matrix_" + output_tag 

            if save:
                plt.savefig(output_fig + '.pdf', format='pdf')
                print("The figure has been saved")

        ax.set_title('Correlation matrix group')# Set the title for each subplot
            
        # output filename
        output_fig=output_dir + "meancorr_matrix_" + output_tag 
        

        # Adjust layout to avoid overlapping
        plt.tight_layout()
        plt.show()

       
        if save==True:
            #plt.savefig(output_fig + '.svg', format='svg')
            plt.savefig(output_fig + '.pdf', format='pdf')
            print("the figure has been saved")


        plt.close()  # Close the figure to release memory
        
    def _plot_matrices(self,heatmap_data,mask,raw,columns,index_raw,index_col,center, norm,cmap,cbar,vmin, vmax,ax,labels):
        
        if index_col:
            heatmap_data = heatmap_data.reindex(index=index_raw, columns=index_col)
        if labels is None:
            labels=""
        
        sns.heatmap(heatmap_data, mask=mask,annot=False, cmap=cmap,  norm=norm,vmin=vmin, vmax=vmax, center=center, cbar=cbar,ax=ax)#,
        #xticklabels=labels.flatten(),yticklabels=labels.flatten())  #Plotting the heatmap
        


    def boxplots(self, df=None, x_data=None, x_order=None, y_data=None, hue=None, hue_order=None,output_dir=None, palette=None, indiv_values=False,indiv_hue=None, indiv_color=None, plot_legend=True, output_tag='', ymin=-1, ymax=1,height=3,aspect=0.8, invers_axes=False,indiv=False, group=False, save=False):
        '''
        Create matrix of correlation boxplots with matching box outline and whisker colors.
        '''

        # Set style and default palette if not provided
        sns.set(style="ticks")
        if palette is None:
            palette = sns.color_palette("pastel")
        if hue is None:
            hue = x_data
            hue_order = x_order
        
        if invers_axes:
            x_data_f=y_data
            y_data_f=x_data
        else:
            x_data_f=x_data
            y_data_f=y_data


        # Create the boxplot

        g = sns.catplot(
                x=x_data_f, 
                y=y_data_f, 
                data=df,  
                kind="box",  
                linewidth=2, 
                palette=palette,  # Use the provided palette
                medianprops=dict(color="white"),  # Set median line color to white
                hue=hue, 
                order=x_order, 
                hue_order=hue_order,
                fliersize=0,  # Remove outliers' markers
                height=height,
                aspect=aspect,
                legend=plot_legend
            )



        # Apply custom outline and whisker colors to match the palette
        for ax in g.axes.flat:
            # Add a horizontal line at y=0
            if invers_axes:
                ax.axvline(0, color='grey', linestyle='--', linewidth=1)
            else:
                ax.axhline(0, color='grey', linestyle='--', linewidth=1)
            # Change whisker colors
            
            

            for i, box in enumerate(ax.patches):  # Access the box patches
                category = df[x_data_f].unique()[i % len(df[x_data_f].unique())]  # Use modulus to loop over categories
                color_index = list(df[x_data_f].unique()).index(category)  # Get the index of the category in the unique list
                color = palette[color_index]  # Use the correct color from the palette
            
                
                # Set the box color and alpha
                box.set_color(color)  # Set box color
                box.set_alpha(0.2)  # Set alpha for the box
                
                whisker_lines = ax.lines[i * 6:i * 6 + 2]  # Whiskers are the first two lines for each box
                for whisker in whisker_lines:
                    whisker.set_color(color)  # Set the whisker color
                    whisker.set_alpha(0.7)  # Set alpha for whiskers

                cap_lines = ax.lines[i * 6 + 2:i * 6 + 4]  # Caps are the next two lines for each box
                for cap in cap_lines:
                    cap.set_color(color)  # Set the cap color
                    cap.set_alpha(0.7)


                # Loop through each box and set outline color
            
                # Get the current category for the box
                category = df[x_data_f].unique()[i % len(df[x_data_f].unique())]  # Use modulus to loop over categories
                color_index = list(df[x_data_f].unique()).index(category)  # Get the index of the category in the unique list
                color = palette[color_index]  # Use the correct color from the palette
                
                # Set the box color
                box.set_color(color)  # Set box color
                
                # Get the bounding box by extracting the vertices of the path
                vertices = box.get_path().vertices
                x_pos = vertices[:, 0].min()  # Minimum x value
                y_pos = vertices[:, 1].min()  # Minimum y value
                width = vertices[:, 0].max() - x_pos  # Width
                height = vertices[:, 1].max() - y_pos  # Height

                # Create a new outline with lower alpha for the edge
                outline = plt.Rectangle(
                    (x_pos, y_pos),  # Position as a tuple
                    width,  # Width
                    height,  # Height
                    fill=False,  # No fill for the outline
                    edgecolor=color,  # Same color as the box
                    lw=1.5,  # Line width
                    alpha=0.7  # Set alpha for transparency of the outline
                )
                ax.add_patch(outline)  # Add the outline to the axis



        # Add individual points if requested
        if indiv_values:
            if indiv_color and indiv_hue:
                sns.stripplot(
                    x=x_data_f, 
                    y=y_data_f, 
                    data=df, 
                    hue=indiv_hue, 
                    hue_order=hue_order,
                    palette=indiv_color,
                    size=5, 
                    linewidth=0.5, 
                    alpha=0.6,jitter=0.25,
                    edgecolor='white'
                )
            else:
                sns.stripplot(
                x=x_data_f, 
                y=y_data_f, 
                data=df, 
                hue=hue, 
                hue_order=hue_order,
                size=4, 
                palette=palette, 
                linewidth=0.5, 
                alpha=0.3,
                edgecolor='white',
                jitter=0.25
            )

        # Set axis labels and formatting
        #g.set_axis_labels(" ", "corr", fontsize=12, fontweight='bold')
        
        if output_tag:
            g.set(title=output_tag)

        if invers_axes:
            g.set(xlim=(ymin, ymax))
        else:
            g.set(ylim=(ymin, ymax))
        sns.despine(offset=30, trim=True)
        if plot_legend:
            g.add_legend()
        else:
            plt.legend([],[], frameon=False)
        
        


        # Save the figure if requested
        if save:
            print(output_dir + output_tag + '.pdf')
            plt.savefig(output_dir + output_tag + '.pdf', dpi=500, transparent=True)

            plt.close()  # Close the figure to release memory
    
    def barplots(self, df=None, x_data=None, x_order=None, y_data=None, hue=None, hue_order=None, output_dir=None, palette=None, indiv_values=False,indiv_hue=False,indiv_color=False,plot_legend=True, output_tag='', ymin=-1, ymax=1,height=3,aspect=0.8, invers_axes=False,indiv=False, group=False, save=False):
        '''
        Create matrix of correlation boxplots with matching box outline and whisker colors.
        '''

        # Set style and default palette if not provided
        sns.set(style="ticks")
        if palette is None:
            palette = sns.color_palette("pastel")
        if hue is None:
            hue = x_data
            hue_order = x_order
        
        if invers_axes:
            x_data_f=y_data
            y_data_f=x_data
        else:
            x_data_f=x_data
            y_data_f=y_data


        # Create the boxplot
        g = sns.catplot(
                x=x_data_f, 
                y=y_data_f, 
                data=df,  
                kind="bar",  
                linewidth=2, 
                palette=palette,  # Use the provided palette
                hue=hue, 
                order=x_order, 
                hue_order=hue_order,
                errorbar=None,
                height=height,
                aspect=aspect
            )


        # Apply custom outline and whisker colors to match the palette
        for ax in g.axes.flat:
            # Add a horizontal line at y=0
            if invers_axes:
                ax.axvline(0, color='grey', linestyle='--', linewidth=1)
            else:
                ax.axhline(0, color='grey', linestyle='--', linewidth=1)
            # Change whisker colors




        # Add individual points if requested
        if indiv_values:
            if indiv_color and indiv_hue:
                sns.stripplot(x=x_data_f, 
                    y=y_data_f, 
                    data=df, 
                    hue=indiv_hue, 
                    hue_order=hue_order,
                    palette=indiv_color,
                    size=4, 
                    linewidth=0.5, 
                    alpha=0.5,
                    edgecolor='white'
                )

            else:
                sns.stripplot(x=x_data_f, 
                y=y_data_f, 
                data=df, 
                hue=hue, 
                hue_order=hue_order,
                size=4, 
                palette=palette, 
                linewidth=0.5, 
                alpha=0.5,
                edgecolor='white'
            )

            


        # Set axis labels and formatting
        #g.set_axis_labels(" ", "corr", fontsize=12, fontweight='bold')
        
        if output_tag:
            g.set(title=output_tag)

        if invers_axes:
            g.set(xlim=(ymin, ymax))
        else:
            g.set(ylim=(ymin, ymax))
        sns.despine(offset=30, trim=True)
        if plot_legend:
            g.add_legend()
        else:
            plt.legend([],[], frameon=False)  
        
        


        # Save the figure if requested
        if save:
            print(output_dir + output_tag + '.pdf')
            plt.savefig(output_dir + output_tag + '.pdf', dpi=500, transparent=True)

            plt.close()  # Close the figure to release memory
     
    def lmplots(self,df=None,x_data=None,x_order=None,y_data=None,order=1,col=None,col_order=None,hue_var=None,hue_color_var=None, hue_order=None,output_dir=None,output_tag=None,color=None, hue_palette=None, indiv_values=True,height=3,aspect=1,ymin=None, ymax=None,xmin=None, xmax=None,xy_plot=False,indiv=False,group=False,save=False):
        '''
        Create a linear model plot with regression lines for each category in 'col'.
        
        Parameters:
        df: DataFrame containing the data
        x_data: Column name for the x-axis data
        x_order: Order of categories for the x-axis
        y_data: Column name for the y-axis data
        order: Order of the polynomial regression line (default is 1 for linear regression)
        col: Column name for the facet grid (optional)
        col_order: Order of categories for the facet grid (optional)
        hue_var: Column name for the hue variable (optional)
        hue_order: Order of categories for the hue variable (optional)
        hue_color_var: Column name for the hue color variable, ex: one color per age (optional)
        output_dir: Directory to save the plot
        output_tag: Tag for the output filename
        color: Color for the regression line and scatter points (default is a single color)
        hue_palette: Color palette for the hue variable (optional)
        indiv_values: If True, individual data points will be plotted (default is True)
        height: Height of each facet plot (default is 3)
        aspect: Aspect ratio of each facet plot (default is 1)
        ymin: Minimum y-axis limit (default is None, which will be calculated from the data)
        ymax: Maximum y-axis limit (default is None, which will be calculated from the data)
        xmin: Minimum x-axis limit (default is None, which will be calculated from the data)
        xmax: Maximum x-axis limit (default is None, which will be calculated from the data)

        '''
        # ----- Set default color if none
        if color is None:
            color = ['#5c5ce0']  # A list with one default color

        # ----- Plot using FacetGrid
        if isinstance(color, list) and len(color) > 1 or hue_palette is not None:
            g = sns.FacetGrid(df, col=col, col_order=col_order, height=height, aspect=aspect, legend_out=False)

            for i, ax in enumerate(g.axes.flat):
                subset = df[df[col] == col_order[i]] if col else df

                # Plot points
                if indiv_values:
                    if hue_color_var:  # Use gradient coloring
                        scatter = ax.scatter(
                            subset[x_data], subset[y_data],
                            c=subset[hue_color_var], cmap=hue_palette,
                            s=40, alpha=0.8, edgecolors='w')
                        sns.regplot(x=subset[x_data], y=subset[y_data],  order=order,color=color[0],
                                        scatter=False, ax=ax, line_kws={'alpha': 0.8})
                        if hue_palette:
                            g.fig.colorbar(scatter, ax=ax, label=hue_color_var)
                    elif hue_var:  # Categorical hue, plot one line per level
                        for j, hue_level in enumerate(subset[hue_var].unique()):
                            subsub = subset[subset[hue_var] == hue_level]
                            ax.scatter(subsub[x_data], subsub[y_data], label=hue_level,color=color[j], alpha=0.6, s=40)
                            sns.regplot(x=x_data, y=y_data, data=subsub, order=order,color=color[j],
                                        scatter=False, ax=ax, line_kws={'alpha': 0.8})
                        ax.legend()
                    else:  # Single color
                        ax.scatter(subset[x_data], subset[y_data], color=color[i], s=40, alpha=0.5)
                        sns.regplot(x=x_data, y=y_data, data=subset, order=order,
                                    scatter=False, ax=ax, line_kws={'color': color[i], 'alpha': 0.8})
                else:
                    # No individual points
                    if hue_var:  # Categorical hue, plot one line per level
                        for j, hue_level in enumerate(subset[hue_var].unique()):
                            subsub = subset[subset[hue_var] == hue_level]
                            sns.regplot(x=x_data, y=y_data, data=subsub, order=order,color=color[j],
                                        scatter=False, ax=ax, line_kws={'alpha': 0.8})
                    else:
                        sns.regplot(x=x_data, y=y_data, data=subset, order=order,
                                scatter=False, ax=ax, line_kws={'color': color[i], 'alpha': 0.8})

        # ----- Simple layout (no hue/color distinction)
        else:
            g = sns.lmplot(x=x_data, y=y_data, hue=hue_var, data=df, order=order,
                        col=col, col_order=col_order, scatter=indiv_values,
                        scatter_kws={'color': color[0]} if indiv_values else None,
                        line_kws={'color': color[0], 'alpha': 0.8},
                        height=height, aspect=aspect, legend=True)


        #------ Set the minimum and maximum limits of the plot
        if xmin is None:
            xmin = df[x_data].min() - 0.5 * (df[x_data].max() - df[x_data].min())
            xmax = df[x_data].max() + 0.5 * (df[x_data].max() - df[x_data].min())
        if ymin is None:
            ymin = df[y_data].min() - 0.5 * (df[y_data].max() - df[y_data].min())
            ymax = df[y_data].max() + 0.5 * (df[y_data].max() - df[y_data].min())
        
        for ax in g.axes.flat:  # In case there are multiple subplots
            ax.axhline(0, color='grey', linestyle='--', linewidth=1)

        #------ Set the axis labels and title
        g.set_axis_labels(x_data, y_data, fontsize=12, fontweight='bold')
        g.add_legend()
        g.set(ylim=(ymin, ymax))
        g.set(xlim=(xmin, xmax))

        if xy_plot:
            x_limits = ax.get_xlim()
            y_limits = ax.get_ylim()
            xy_min = min(x_limits[0], y_limits[0])
            xy_max = max(x_limits[1], y_limits[1])
            ax.plot([xy_min, xy_max], [xy_min, xy_max], color='grey', linestyle='dotted', linewidth=1)
  
        sns.despine(offset=10, trim=True)

        g.fig.subplots_adjust(top=0.88)  # Adjust if needed
        g.fig.suptitle(output_tag, y=1.05)

        plt.show()
        
        if save and output_dir:
            g.savefig(f'{output_dir}/{output_tag}.pdf', bbox_inches='tight')

            plt.close()  # Close the figure to release memory
    
    def regplots(self,df=None,x_data=None,y_data=None, reg_color="#0896ae",x_color=None,y_color=None,output_f=None,height=5,aspect=4,hlines=True,ymin=None, ymax=None,xmin=None, xmax=None,save=False):
        """
        Create a regression plot with marginal histograms.
        df: DataFrame containing the data
        x_data: Column name for the x-axis data
        y_data: Column name for the y-axis data
        reg_color: Color for the regression line and scatter points
        x_color: Color for the x-axis marginal histogram (default is reg_color)
        y_color: Color for the y-axis marginal histogram (default is reg_color)
        output_dir: Directory to save the plot
        output_tag: Tag for the output filename
        """
        
        #--------- Set the plot ---------
        g = sns.jointplot(
            data=df,
            x=x_data, y=y_data,color=reg_color,
            kind="reg"
        )
        
        #--------- Set size and limits of the plot ---------
        g.fig.set_size_inches(height, aspect)
        if xmin is None:
            xmin = df[x_data].min() - 0.05 * (df[x_data].max() - df[x_data].min())
            xmax = df[x_data].max() + 0.05 * (df[x_data].max() - df[x_data].min())
        if ymin is None:
            ymin = df[y_data].min() - 0.05 * (df[y_data].max() - df[y_data].min())
            ymax = df[y_data].max() + 0.05 * (df[y_data].max() - df[y_data].min())
        g.ax_joint.set(xlim=(xmin, xmax), ylim=(ymin, ymax))

        #g.ax_joint.set_xlim(-0.05, 0.4)  # example limits for x-axis

        #--------- Set the color setting ---------
        if x_color is None:
            x_color = reg_color
        if y_color is None:
            y_color = reg_color

        for bar in g.ax_marg_x.patches:
            bar.set_facecolor(x_color)
            bar.set_alpha(0.9)
            bar.set_edgecolor("#FFFFFF")
        
        for bar in g.ax_marg_y.patches:
            bar.set_facecolor(y_color)
            bar.set_alpha(0.9)
            bar.set_edgecolor("#FFFFFF")

        for line in g.ax_marg_x.lines:
            line.set_color(x_color)
            line.set_alpha(0.5)

        for line in g.ax_marg_y.lines:
            line.set_color(y_color)
            line.set_alpha(0.5)
        
        if hlines:
            g.ax_joint.axhline(y=0, color="grey", linestyle='--', linewidth=1)
                    

        plt.tight_layout()
        plt.show()

        if save :
            output_fig = f'{output_f}.pdf'
            g.savefig(output_fig, bbox_inches='tight', dpi=300)
            print(f"The figure has been saved to {output_fig}")
            plt.close()
    
    def lineplots(self,df=None,x_data=None,x_order=None,y_data=None,hue=None,hue_order=None, output_dir=None,palette=None, indiv_values=False,output_tag='',ymin=-1, ymax=1,hsize=6,wsize=6,indiv=False,group=False,tag='',save=False):

        # 1. Create a palette based on the number of groups ____________________
        sns.set(style="ticks")#,  font='sans-serif')
        print(x_data)
        if hue:
            unique_groups = df[hue].unique() # select the number of groups
            if palette==None:
                palette = sns.color_palette("husl", len(unique_groups))  # Create a default palette
            # Create a dictionary to map groups to dynamically generated colors
            group_palette = dict(zip(unique_groups, palette))

        # 2 Clean the dataframe and calulate statistics ______________________
        # Remove NaN values from MEAN(area)
        df_clean = df.dropna(subset=[y_data])
        if hue:
            df_grouped = df_clean.groupby([hue, x_data]).agg(
            mean_area=(y_data, 'mean'),
            std_area=(y_data, 'std'),
            count=(y_data, 'count')  # Count the number of individuals per group and VertLevel
        ).reset_index()
        else:
            df_grouped = df_clean.groupby([x_data]).agg(
            mean_area=(y_data, 'mean'),
            std_area=(y_data, 'std'),  # Calculate the standard deviation of the area
            count=(y_data, 'count')  # Count the number of individuals per group and VertLevel
        ).reset_index()

        df_grouped['sem_area'] = df_grouped['std_area'] / np.sqrt(df_grouped['count'])# Calculate the standard error of the mean (SEM)

        # 3. Predefine ymin and ymax for the y-axis range______________________
        if ymin is None:
            ymin = df_grouped['mean_area'].min() - 5  # Optional padding below the min value
        if ymax is None:
            ymax = df_grouped['mean_area'].max() + 5  # Optional padding above the max value

        # Plot the mean with the standard error around it
        plt.figure(figsize=(wsize,hsize))  # Set the figure size

        # Loop over each group to plot mean with SEM as shaded area
        if hue:
            for group_nb, group in enumerate(hue_order):

                group_data = df_grouped[df_grouped[hue] == group]
                # Plot the line for y with the specified color palette
                sns.lineplot(data=group_data, 
                    x=x_data, y='mean_area', 
                    label=group, 
                    marker="o", 
                    color=palette[group_nb])  # Use the custom palette for group color
        
                # Plot the shaded area representing the standard error (SEM)
                plt.fill_between(group_data[x_data], 
                        group_data['mean_area'] - group_data['sem_area'], 
                        group_data['mean_area'] + group_data['sem_area'], 
                        color=palette[group_nb],  # Match the fill color to the line color
                        alpha=0.2)  # alpha=0.2 makes the shaded area transparent
        else:
            sns.lineplot(data=df_grouped, x=x_data, y='mean_area', marker="o", color=palette)
            plt.fill_between(df_grouped[x_data],
                df_grouped['mean_area'] - df_grouped['sem_area'],
                df_grouped['mean_area'] + df_grouped['sem_area'],
                color=palette, alpha=0.2)

        # Add titles and labels
        sns.despine(offset=10,trim=True)
        if hue:
            plt.title(tag + ' plot: ' +y_data + ' by ' +x_data+ ' for Different ' + hue)
        else:
            plt.title(tag + ' plot: ' +y_data + ' by ' +x_data)
        plt.xlabel(x_data)
        plt.ylabel(y_data)
        plt.ylim(ymin, ymax)

        # Show plot
        plt.legend(title=hue)
        #plt.show()

        if save==True:
            output_fig=output_dir + output_tag
            plt.savefig(output_fig + '.pdf',dpi=500,transparent=True)
            plt.close()  # Close the figure to release memory

    def plot_radar_chart(self,df=None,metric="corr",categories=None, hue=None,hue_order=None,y_max=None,y_min=None,colors='blue',output_tag="",output_dir=None,save=False):
        
        '''
        Function to plot radar chart,
        df: dataframe
        metric: name of the column that contains the values to plot
        categories: name of the columns that contains the categories
        hue: name of the column that contains the hue
        hue_order: specify the order of the hue
        y_max: value
        color: color of the radar chart
        output_dir: output directory
        output_tag: tag of the output file
        save: bool
        '''

        #Check if df and categories variable exists
        if df is None:
            raise ValueError('Please provide a dataframe')
        if categories is None:
            raise ValueError('Please provide the categories to plot')
        
        # Create a list of categories
        categories = df[categories].unique()
        N = len(categories)  # Number of categories

        
        if hue:
            values={}
            sub_groups = df[hue].unique()
            #colors = itertools.cycle(color)  # Cycle through colors if too few

            if hue_order:
                hue_order = [h for h in hue_order if h in sub_groups]# Validate that hue_order is a subset of actual unique groups
            else:
                hue_order = sub_groups

            for sub_group in hue_order:
                df_group = df[df[hue] == sub_group]
                values[sub_group] = df_group[metric].values
                values[sub_group] = np.concatenate((values[sub_group], [values[sub_group][0]]))  # Add the first value to the end to close the circle
            
        else:
            values = df[metric].values
            values = np.concatenate((values, [values[0]]))  # Add the first value to the end to close the circle
           
        
        max_value = max([max(values[sub_group]) for sub_group in hue_order]) if hue else max(values) # Get the maximum value for the y-axis
        min_value = min([min(values[sub_group]) for sub_group in hue_order]) if hue else min(values) # Get the maximum value for the y-axis

        # Create a list of angles
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist() # Set the angles for each category
        angles = np.concatenate((angles, [angles[0]])) # Add the first angle to the end to close the circle

        #Plot the radar chart
        fig, ax = plt.subplots(figsize=(3, 3), subplot_kw=dict(polar=True))  # Create a polar plot

        if hue:
            for sub_group, col in zip(hue_order, colors):
                ax.fill(angles, values[sub_group], color=col, alpha=0.1)  # Fill the area
                ax.plot(angles, values[sub_group], color=col, linewidth=1)
                ax.plot([], [], color=col, label=sub_group)  # Add legend entries
            #ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))  # Legend

        else:
            ax.fill(angles, values, color=colors[0], alpha=0.1)
            ax.plot(angles, values, color=colors[0], linewidth=1)
        
        ax.set_xticks(angles[:-1])  # Set the x-ticks to the angles
        ax.set_xticklabels(categories,fontsize=10)  # Set the x-tick labels to the categories
        if y_max is None:
            y_max=max_value* 1.1

        if y_min is None:
            y_min=min_value if min_value<0 else 0


        ax.set_ylim(y_min, y_max) # Set y-axis limits dynamically
        #ax.set_facecolor('lightgray') #color of the circle
        ax.grid(color='lightgray', linestyle='-', linewidth=0.5, alpha=0.2) 
        ax.set_yticklabels([])  # Remove the y-ticks

        # To add a thicker border around the plot (contour)
        ax.spines['polar'].set_visible(True)  # Ensure the contour (spine) is visible
        ax.spines['polar'].set_color('lightgray')  # Change the color of the contour to red
        ax.spines['polar'].set_linewidth(0.5)  # Set the width of the contour line


        # Save the figure if requested
        plt.show()
        
        if save and output_dir:
            fig.savefig(f'{output_dir}/{output_tag}.pdf', bbox_inches='tight')

            plt.close()  # Close the figure to release memory

    def donutplots(self, y_data=None, labels=None, palette=None,output_dir=None,  output_tag='', save=False):
        '''
        Create a donut plot 
        y_data: name of the column that contains the values to plot
        labels: name of the column that contains the labels, if None no text will be displayed
        palette: list of colors to use for the plot, if None a default palette will be used
        output_dir: output directory
        output_tag: tag of the output file
        save: bool, if True the plot will be saved in the output directory
        '''
        # Check if df and y_data variable exists
        if y_data is None:
            raise ValueError('Please provide the y_data to plot')

        fig, ax = plt.subplots(figsize=(2, 2))
        wedges, texts = ax.pie(y_data,
        labels=labels,
        startangle=90,
        colors=palette,
        counterclock=False,
        wedgeprops=dict(width=0.3)  )

        plt.show()
        if save:
            output_fig = output_dir + output_tag
            fig.savefig(output_fig + '.pdf', dpi=500, transparent=True)
            plt.close(fig)


    
 # ======= Brain ========
class Plot_brain:
    def __init__(self, config_file):
        with open(config_file) as config_f:
            self.config = json.load(config_f) # load config info
        self.config_file=config_file

    def plot_colormap(colors=None,plot_colormap=True,redo=False,verbose=True):
        '''
        Plot your colormap
        Attributes
        ----------
        colors: list of color in html format (i.g. ["#1a04a4",'#0070ff','#07f6e0', "#9bff00",'#e8f703', '#fa9c03', '#ff3a00'])
        '''

        # Create a ListedColormap from the specified colors
        discretized_colormap = ListedColormap(colors)
       
        # Plot a colorbar to visualize the colormap
        if plot_colormap==True:
            plt.figure(figsize=(3,2))
            plt.imshow(np.arange(1, 8).reshape(1, -1), cmap=discretized_colormap, aspect='auto', vmin=1, vmax=7)
            plt.colorbar(ticks=np.arange(1, 8))
            plt.show() 
        
        return discretized_colormap

    def plot_3D(self, i_img=None,hemi_view=["lh","rh"],face_view=["lateral"],vmin=None,vmax=None,threshold=1e-6, mask_img=None,colormap='hot', tag="",output_dir=None, save_results=False):
        '''
        This function help to plot functional 3D maps on a render surface 
        Two nilearn function are used, see details here:
        https://nilearn.github.io/dev/modules/generated/nilearn.surface.vol_to_surf.html
        https://nilearn.github.io/dev/modules/generated/nilearn.plotting.plot_surf_roi.html
        
        to do:
        - Add option to import directly a volume
        - Add new value possibilities for view_per_line
        
        Attributes
        ----------
        i_img <filename>: filename of the input 3D volume image (overlay) should be specify, default: False
        hemi_view <str>: hemisphere of the brain to display, one of the three options should be specified: ["lh","rh"] or ["lh"] or ["rh"]
        face_view <str> : must be a string in: "lateral","medial","dorsal", "ventral","anterior" or "posterior" multiple views can be specified simultaneously (e.g., ["lateral","medial"])
        vmin <float>, optional, Lower bound of the colormap. If None, the min of the image is used.
        vmax <float>, optional, upper bound of the colormap. If None, the min of the image is used. 
        threshold <int> or None optional. If None is given, the image is not thresholded. If a number is given, it is used to threshold the image: values below the threshold are plotted as transparent. If “auto” is given, the threshold is determined magically by analysis of the image. Default=1e-6.
        colormap <str> : specify a colormap for the plotting, default: 'automn'
        tag <str> : specify a tag name for the output plot filenames, default: '' 
        output_dir <directory name>, optional, set the output directory if None, the i_img directory will be used
        save_results <boolean>, optional, set True to save the results
        
        '''
        
        if i_img==None:
            raise Warning("Please provide the filename of the overlay volume (ex: i_img='/my_dir/my_func_img.nii.gz')")
        if output_dir==None:
            output_dir=os.path.dirname(i_img) + "/plots/"
        
        if save_results:
            if not os.path.exists(output_dir):
                os.mkdir(output_dir) # create output folder
                os.mkdir(output_dir + "/tmp/") # create a temporary folder
     
        # 1. Select surface image for background --------------------------------------------------------
        surface_dir = self.config['project_dir'] + self.config['templates']['surface_dir']
        
        img_surf={}
        for hemi in hemi_view:
            #2. Transform volume into surface image --------------------------------------------------------------------
            # include a mask is there are 0 values that you don't want to include in the interpolation
            img_surf[hemi]=surface.vol_to_surf(i_img,surface_dir+ hemi + ".pial",radius=0, 
                                     interpolation='nearest', kind='line', n_samples=10, mask_img=mask_img, depth=None)
  

            #3. Plot surface image --------------------------------------------------------------------
            side = "left" if hemi == "lh" else "right"
            for face in face_view:
                colorbar=True if face == face_view[-1] else False
                plot=plotting.plot_surf_roi(surface_dir+ hemi +".inflated", roi_map=img_surf[hemi],
                                            cmap=colormap, colorbar=colorbar,#mask_img=mask_img,
                                            hemi=side, view=face,vmin=vmin,vmax=vmax,threshold=int(threshold),
                                            bg_map=surface_dir + hemi +".sulc",darkness=.7)

                if save_results:
                    
                    # Save each plot individually
                    plot.savefig(os.path.join(output_dir + "/tmp/", f'plot_{tag}_{hemi}_{face}.png'),dpi=150)
                    plt.close()
                plt.show()

        if save_results:
            # Compute number of columns/rows and prepare subplots accordingly
            view_per_line= len(face_view)

            total_rows = ((len(face_view)*len(hemi_view))//view_per_line)

      
            fig, axs = plt.subplots(total_rows, view_per_line, figsize=(10*view_per_line, 8*total_rows))
                    
            for col, face in enumerate(face_view):
                for row, hemi in enumerate(hemi_view):
                    img = plt.imread(os.path.join(output_dir + "/tmp/", f'plot_{tag}_{hemi}_{face}.png'))

                    if len(face_view)==1:
                        axs[row].imshow(img)
                        axs[row].axis('off')
                    else:
                        axs[row,col].imshow(img)
                        axs[row,col].axis('off')
        
            # Save the combined figure
            combined_save_path = os.path.join(output_dir, tag+".pdf")
            plt.savefig(combined_save_path, bbox_inches='tight')
            plt.show()
            
            #remove temporary files
            for hemi in hemi_view:
                for face in face_view:
                    os.remove(os.path.join(output_dir + "/tmp/", f'plot_{tag}_{hemi}_{face}.png'))


    
     # ======= Spinal Cord ========
class Plot_sc:
    def __init__(self, config_file,analysis="icaps",method="max intensity"):
        '''
        This function help to plot functional 3D maps on PAM50 template
        Parameters
        ----------
        config_file: str
            Path to the configuration file (JSON format)
        analysis: str
            Type of analysis to be performed (default: "icaps")
        method: str
            Method to determine spinal levels (default: "max intensity")
        '''


        with open(config_file) as config_f:
            self.config = json.load(config_f) # load config info

        self.i_img=glob.glob(self.config["project_dir"] + self.config[analysis]["analysis_dir"]["spinalcord"] +"/iCAPs_z.nii")[0]
        self.data=nib.load(self.i_img).get_fdata()
        self.data_sorted = {} # To store the data sorted WITHIN dataset (e.g., rostrocaudally)
        self.spinal_levels_sorted={} 

        ##########
        # sort 4d icaps maps in rostrocaudal plane:
        max_z = []; 
        for i in range(0,self.data.shape[3]):
            max_z.append(int(np.where(self.data == np.nanmax(self.data[:,:,:,i]))[2][0]))  # take the first max in z direction      
        sort_index = np.argsort(max_z)
        sort_index= sort_index[::-1] # Invert direction to go from up to low

        self.data_sorted = self.data[:,:,:,sort_index] 

        ###########
        #Match maps to corresponding spinal levels
        levels_list = sorted(glob.glob(self.config['project_dir'] + self.config['templates']['spinalcord']["levels_path"] + 'spinal_level_*.nii.gz')) #  Sorted is used to make sure files are listed from low to high number (i.e., rostro-caudally)
        # Prepare structures
        levels_data = np.zeros((self.data.shape[0],self.data.shape[1],self.data.shape[2],len(levels_list))) # To store spinal levels, based on size of 4D data (corresponding to template) & number of spinal levels in template
        spinal_levels = np.zeros(self.data.shape[3],dtype='int') # To store corresponding spinal levels

         # Loop through levels & store data
        for lvl in range(0,len(levels_list)):
            level_img = nib.load(levels_list[lvl])
            levels_data[:,:,:,lvl] = level_img.get_fdata()


        if method=="CoM":
            map_masked = np.where(self.data > 1.5, self.data, 0) # IMPORTANT NOTE: here, a low threshold at 1.5 is used, as the goal is to have rough maps to match to levels
            CoM = np.zeros(map_masked.shape[3],dtype='int')
            for i in range(0,self.data.shape[3]):
                _,_,CoM[i]=center_of_mass(map_masked[:,:,:,i])
                # Take this point for each level (we focus on rostrocaudal position and take center of FOV for the other dimensions)
                level_vals = levels_data[levels_data.shape[0]//2,levels_data.shape[1]//2,CoM[i],:]
                
                spinal_levels[i] = np.argsort(level_vals)[-1] if np.sum(level_vals) !=0 else -1 # Take level with maximum values (if no match, use -1)
            
        elif method=="max intensity":
            # For each map, find rostrocaudal position of point with maximum intensity
            max_intensity = np.zeros(self.data.shape[3],dtype='int')
            
            for i in range(0,self.data.shape[3]):
                max_size=np.where(self.data == np.nanmax(self.data[:,:,:,i]))[2].size
                if max_size>1:
                    max_intensity[i] = np.where(self.data == np.nanmax(self.data[:,:,:,i]))[2][int(max_size/2)] # take the middle max if there are mainy
                else:
                    max_intensity[i] = np.where(self.data == np.nanmax(self.data[:,:,:,i]))[2]
                
                #print(max_intensity)
                # Take this point for each level (we focus on rostrocaudal position and take center of FOV for the other dimensions)
                level_vals = levels_data[levels_data.shape[0]//2,levels_data.shape[1]//2,max_intensity[i],:]
                spinal_levels[i] = np.argsort(level_vals)[-1] if np.sum(level_vals) !=0 else -1 # Take level with maximum values (if no match, use -1)

        self.spinal_levels_sorted = spinal_levels[sort_index]
        

    def icaps_plot(self, n_k=7,k_per_line=None, lthresh=None, uthresh=4.0, perc_thresh=90, template=None, centering_method='max', plot_mip=False, show_spinal_levels=False, output_dir=None,colormap='autumn',alpha = 0.7, save_results=False):
        ''' Plot components overlaid on PAM50 template (coronal and axial views are shown)
        
        Inputs
        ----------
        i_img: filename
            4D input image containing the icap components
        n_k: float
            Number of K
        k_per_line : str
            Number of maps to display per line (default = will be set to total number of 4th dimension in the 4D image)
        lthresh : float
            Lower threshold value to display the maps 
        uthresh : float
            Upper threshold value to display the maps (default = 4.0)
        template : str
            To change the background if needed (default = None)
            If None, the template image defined in the config file is used
        centering_method : str
            Method to center display in the anterio-posterior direction (default = 'max')
                'max' to center based on voxel with maximum activity
                'middle' to center in the middle of the volume
        plot_mip : boolean
            If set to True, we plot the Maximum Intensity Projection (default = False)
        show_spinal_levels : boolean
            Defines whether spinal levels are displayed or not (default = False)
        colormap : str
            Defines colormap used to plot the maps if one dataset (default = 'autumn')
        save_results : boolean
            Set to True to save figure (default = False)'''
        
        print("The plotting will be displayed in neurological orientation (Left > Right)")

        if plot_mip and centering_method != "middle":
            print("When using the maximum intensity projection, centering method is set to 'middle'")
            centering_method = "middle" # "force" centering method

        
        
        # By default, use a single line for all 3D maps, otherwise use provided value
        if (k_per_line is not None and k_per_line <= n_k) or k_per_line is None:
            k_per_line = n_k if k_per_line is None else k_per_line
        else:
            raise(Exception('Number of maps per line should be inferior or equal to the total number of maps.'))
        
        # Load template image for background
        template_img = nib.load(self.config['project_dir'] + self.config['templates']['spinalcord']['t2']) if template is None else nib.load(template)
        template_data = template_img.get_fdata()
        map_masked = {}
        
        if show_spinal_levels == True: # Load levels if needed
            # Find list of spinal levels to consider (defined in config)
            levels_list = sorted(glob.glob(self.config['project_dir'] +self.config['templates']['spinalcord']["levels_path"] + 'spinal_level_*.nii.gz')) # Sorted is used to make sure files are listed from low to high number (i.e., rostro-caudally)
            levels_data = np.zeros((self.data.shape[0],self.data.shape[1],self.data.shape[2],len(levels_list))) # To store spinal levels, based on size of 4D map data (corresponding to template) & number of spinal levels in template
            
            # Loop through levels & store data
            for lvl in range(0,len(levels_list)):
                level_img = nib.load(levels_list[lvl])
                levels_data[:,:,:,lvl] = level_img.get_fdata()
                # Mask level data to use as overlays
                levels_data = np.where(levels_data > 0, levels_data, np.nan)      
        
        # To mask maps, values below threshold are replaced by NaN
        map_masked = np.where(self.data_sorted > lthresh, self.data_sorted, np.nan) # Same if only one, no need to take the matched version
        # Compute number of columns/rows and prepare subplots accordingly 
        total_rows = int(np.ceil(n_k/k_per_line)*2) if n_k > k_per_line else 2
        _, axs = plt.subplots(nrows=total_rows,ncols=k_per_line,figsize=(2*k_per_line, 4*total_rows))
        plt.axis('off')

        for i in range(0,n_k):
            # Column is the same for coronal & axial views
            col = i%k_per_line
            # Draw coronal views
            row_coronal = 0 if i<k_per_line else (i//k_per_line-1)*2+2
            axs[row_coronal,col].axis('off')
            
                        
            if centering_method == 'max':
                max_size=np.where(map_masked == np.nanmax(map_masked[:,:,:,i]))[1].size
                if max_size>1:
                    max_y = int(np.where(map_masked== np.nanmax(map_masked[:,:,:,i]))[1][0]) # take the first max if there are mainy
                else:
                    max_y = int(np.where(map_masked == np.nanmax(map_masked[:,:,:,i]))[1])
            elif centering_method == 'middle':
                max_y = template_data.shape[1]//2
                max_y = 26
            else:
                raise(Exception(f'"{centering_method}" is not a supported centering method.'))
                    
            # Show template as background
            axs[row_coronal,col].imshow(np.rot90(template_data[:,max_y,:].T,2),cmap='gray');
                
            # Show spinal levels
            if show_spinal_levels == True:
                axs[row_coronal,col].imshow(np.rot90(levels_data[:,max_y,:,self.spinal_levels_sorted[i]]),cmap='gray_r',alpha=0.8)
            # Show components

            if plot_mip:
                # Compute projection
                mip = np.nansum(map_masked[:,:,:,i].T,axis=1)
                # Threshold
                mip = np.where(mip > lthresh, mip, np.nan)
                axs[row_coronal,col].imshow(np.rot90(mip,2),vmin=lthresh, vmax=uthresh,cmap=colormap,alpha=alpha)
            else:
                axs[row_coronal,col].imshow(np.rot90(map_masked[:,max_y,:,i].T,2),vmin=lthresh, vmax=uthresh,cmap=colormap,alpha=alpha)
            

            # Draw axial views
            row_axial = 1 if i<k_per_line else (i//k_per_line-1)*2+3
            axs[row_axial,col].axis('off');
            
            max_size=np.where(map_masked== np.nanmax(map_masked[:,:,:,i]))[2].size
            if max_size>1:
                max_z = int(np.where(map_masked == np.nanmax(map_masked[:,:,:,i]))[2][int(max_size/2)]) # take the midle max if there are mainy
            else:
                max_z = int(np.where(map_masked == np.nanmax(map_masked[:,:,:,i]))[2])
            
            # Show template as background
            axs[row_axial,col].imshow(np.rot90(template_data[:,:,max_z].T,2),cmap='gray');
            
            # Show components
            axs[row_axial,col].imshow(np.rot90(map_masked[:,:,max_z,i].T,2),vmin=lthresh, vmax=uthresh,cmap=colormap,alpha=alpha)
                
            # To "zoom" on the spinal cord, we adapt the x and y lims
            #axs[row_axial,col].set_xlim([map_masked[main_dataset].shape[0]*0.2,map_masked[main_dataset].shape[0]*0.8])
            #axs[row_axial,col].set_ylim([map_masked[main_dataset].shape[1]*0.2,map_masked[main_dataset].shape[1]*0.8])
            axs[row_axial,col].set_anchor('N')

        # If option is set, save results as a png
        if save_results == True:
            if output_dir==None:
                output_dir=os.path.dirname(self.i_img)
            print(output_dir + "/iCAPs_K" +str(n_k)+'_thr' + str(lthresh)+ 'to' + str(uthresh) + '.pdf')
            plt.savefig(output_dir +  "/iCAPs_K" +str(n_k)+'_thr' + str(lthresh)+ 'to' + str(uthresh) + '.pdf')
    