# Basic data manipulation and viewing libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from scipy.stats import poisson
import tqdm 

# Torch
import torch
from torch.utils.data import DataLoader

# System imports
import sys

# TROT PATHS
sys.path.append('..')
sys.path.append('../Trot')
sys.path.append('../Data')
from Trot.Tsallis import TROT, q_log
from Trot.Generators import euc_costs, real_euc_costs  

# ML4NW data paths
sys.path.append('/afs/desy.de/user/s/sjiggins/HiDA/ML4NW/JupyterHub/ml4nw/ToyModels/CamelFunction/')
sys.path.append('/afs/desy.de/user/s/sjiggins/HiDA/ML4NW/JupyterHub/ml4nw/ToyModels/')
sys.path.append('/afs/desy.de/user/s/sjiggins/HiDA/ML4NW/JupyterHub/ml4nw/')
import utils
import equations

# Function for getting data for plotting
@torch.no_grad()
def get_plot_data(loader):
    temp_x = []
    temp_w = []
    t = tqdm.tqdm(enumerate(loader), total=len(loader))
    for i, batch in t:
        temp_x.append(batch[0])
        temp_w.append(batch[1])
        t.refresh()  # to show immediately the update
    return torch.cat(temp_x).numpy().flatten(), torch.cat(temp_w).numpy().flatten()

# Function for batched samples - input domain
def get_x_i(batch_list, ix):
    x_batch_list = []
    w_batch_list = []
    for sample in batch_list:
        x_batch_list.append(sample[0])
        w_batch_list.append(sample[2])
    x_batch = torch.stack(x_batch_list)[:,ix]
    w_batch = torch.stack(w_batch_list)
    return x_batch, w_batch

# Function for batched samples - radial domain
def get_r(batch_list):
    x_batch_list = []
    w_batch_list = []
    for sample in batch_list:
        x_batch_list.append(sample[0])
        w_batch_list.append(sample[2])
    x_batch = torch.stack(x_batch_list)
    r_batch = np.sqrt(x_batch[:,0]**2 + x_batch[:,1]**2).reshape(-1, 1)
    w_batch = torch.stack(w_batch_list)
    return r_batch, w_batch

# Function for extracting re-weighting function from classifier or optimal classifier
@torch.no_grad()
def get_r_hats(model, loader, X_scaler=None, weight_norm=1, mix=False, leave=False):
    if type(model) is not types.FunctionType:
        model.eval()
        if mix is True:
            loader.collate_fn = lambda batch: utils.preprocessing.prep_inputs_for_training_mix(batch, 
                                                                                               weight_norm=weight_norm)
        else:
            loader.collate_fn = lambda batch: utils.preprocessing.prep_inputs_for_training(batch, X_scaler, 
                                                                                           weight_norm=weight_norm)
    else:
        loader.collate_fn = lambda batch: utils.preprocessing.prep_inputs_for_density(batch, 
                                                                                      weight_norm=weight_norm)

    r_hat_list = []
    t = tqdm(enumerate(loader), total=len(loader), leave=leave)
    for i, batch in t:
        if type(model) is not types.FunctionType:
            x = batch[0].to(DEVICE)
        else:
            x = batch[0].to('cpu')
        batch_output = model(x)
        r_hat = batch_output / (1 - batch_output)
        r_hat_list.append(r_hat)
        t.refresh()  # to show immediately the update

    return torch.cat(r_hat_list).cpu().numpy().flatten()

# Function for plotting the optimal transport solution
def Plot(q, l, 
         nq, nl, n,  
         P, hist_x0, hist_x1, edges,
         file_name = 'TROT_image'):

    # Now generate a figure of the various marginals and transport matrices
    fig = plt.figure(figsize=(8, 8))
    
    # Generate the outter grid of the plots
    outer_grid = gridspec.GridSpec(2, 2, width_ratios=[1,5], height_ratios=[1,5])
    outer_grid.update(wspace=0.01, hspace=0.01)
    # gridspec inside gridspec
    outer_joint = gridspec.GridSpecFromSubplotSpec(nq,nl, subplot_spec=outer_grid[1,1],wspace=0.02, hspace=0.02)
    outer_row_marg = gridspec.GridSpecFromSubplotSpec(nq,1, subplot_spec=outer_grid[1,0],wspace=0.02, hspace=0.02)
    outer_col_marg = gridspec.GridSpecFromSubplotSpec(1,nl, subplot_spec=outer_grid[0,1],wspace=0.02, hspace=0.02)

    #  -> matrices - P
    for b in range(nl):
        for a in range (nq):
            ax = plt.Subplot(fig, outer_joint[a,b])
            #ax.imshow(P[nl*a + b], origin='upper', interpolation = None, aspect = 'auto', cmap = 'Greys')
            ax.imshow(P[nl*a + b], origin='upper', interpolation = None, aspect = 'auto', cmap = 'plasma')#, norm='log')
            #rect = Rectangle((0, 0), n-1, n-1, fc='none', ec='black')     
            #rect.set_width(0.8)
            #rect.set_bounds(0,0,n-1,n-1)
            #ax.add_patch(rect)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
            #ax.set_facecolor('white')
            #print(f'W_{q[a],l[b]} = {np.sum(P[nl*a + b]*euc_costs(n-1,n-1))}')
            
    for i in range(nq):
        ax_row = plt.Subplot(fig,outer_row_marg[i], sharey = ax)
        #ax_row.plot(1-hist_x1, np.arange(0, np.shape(edges)[0]), color='orange' )
        ax_row.plot(hist_x1, np.arange(0, np.shape(edges)[0]), color='orange' )
        ax_row.plot(np.zeros(np.shape(edges)[0]), np.arange(0, np.shape(edges)[0]), 
                    color='grey', linestyle='dashed' )
        fig.add_subplot(ax_row)
        
        ax_row.axes.get_xaxis().set_visible(False)
        ax_row.axes.get_yaxis().set_visible(False)
        bottom, height = .25, .5
        top = bottom + height
        ax_row.text(-0.05, 0.5*(bottom+top), 'q = %.2f' % q[i], horizontalalignment='right', verticalalignment='center', rotation='vertical',transform=ax_row.transAxes, fontsize='medium')
        
        ax_row.set_facecolor('white')
    
    for j in range(nl):
        ax_col = plt.Subplot(fig,outer_col_marg[j], sharex = ax)
        ax_col.plot(np.arange(0, np.shape(edges)[0]), hist_x0)
        ax_col.plot(np.arange(0, np.shape(edges)[0]), np.zeros(np.shape(edges)[0]), 
                    color='grey', linestyle='dashed' )
        fig.add_subplot(ax_col)    
        bottom, height = .25, .5
        ax_col.axes.get_xaxis().set_visible(False)
        ax_col.axes.get_yaxis().set_visible(False)
        ax_col.set_title(r'$\lambda$'+' = {0}'.format(l[j]),fontsize='medium')
        ax_col.set_facecolor('white')
        
    #fig.show()
    plt.savefig(f'{file_name}.pdf', format='pdf')


# Function definition for main
def main():

    
    # Details of marginals etc... for TROT
    #   -> Number of bins in pdfs
    n = 50
    #   -> q-parameter of Tsallis entropy
    #q = [2.0, 1.0, 0.5] #np.arange(0.5,2.5,0.5)
    #q = [1.0, 0.95, 0.5]
    q = [1.0, 0.5]
    #   -> Regularisation factor for entropic regularisation term 
    #      in minimisation objective
    #l = [50, 25, 10, 1, 0.1]
    #l = [100, 50, 25, 10, 1, 0.1]
    #l = [50, 10, 5, 1,] # main
    l = [100, 50]
    
    # File locations
    source_file = '/afs/desy.de/user/s/sjiggins/HiDA/ML4NW/JupyterHub/ml4nw/ToyModels/CamelFunction/base_gaussian_distribution_mc_data'
    target_file = '/afs/desy.de/user/s/sjiggins/HiDA/ML4NW/JupyterHub/ml4nw/ToyModels/CamelFunction/target_negative_camel_distribution_mc_data'

    # Load numpy data sets
    test_base_dataset = utils.preprocessing.Dataset(source_file + "_test.npy", 0)
    test_target_dataset = utils.preprocessing.Dataset(target_file + "_test.npy", 1)

    # Pre-process data
    test_base_dataset.process()
    test_target_dataset.process()

    # Create the dataloaders using torch base methods
    batch_size = int(2**12)
    test_nominal_loader = DataLoader(utils.preprocessing.CombinedDataset(test_base_dataset),
                                     batch_size=batch_size, shuffle=False)
    test_target_loader  = DataLoader(utils.preprocessing.CombinedDataset(test_target_dataset), 
                                     batch_size=batch_size, shuffle=False)

    # Load any re-weighitng methods - NNs or optimal classifiers
    #   -> Source distribution of two gaussians (one positive and one negative)
    source_mixture_coef = (4, -1)
    source_scales = (2.5, 2.5)
    #target_mixture_coef = (2, -1)
    target_mixture_coefs = [ (4/i, -1) for i in reversed(range(2,53))]
    #target_scales = (2, 1.42)
    target_scales = [(2, 1.42) for i in range(2,53) ]


    #    ->   Get the analytical optimal classifier
    s_optimal = []
    for coef_set, scale_set in zip(target_mixture_coefs, target_scales):
        s_optimal.append(equations.optimal_binary_classifier(source_scales[0], 
                                                             coef_set, 
                                                             scale_set))
        

    #quit()
    # Create the plots
    kwargs={'legend.title_fontsize': 18, 
            'font.size'            : 14, 
            'axes.titlesize'       : 20, 
            'axes.labelsize'       : 16,
            'figure.titlesize'     : 20, 
            'ytick.labelsize'      : 12}
    #for i in range(2):
    #    test_nominal_loader.collate_fn = lambda batch: get_x_i(batch, i)
    #    test_target_loader.collate_fn = lambda batch: get_x_i(batch, i)
    #
    #    test_nominal_xi = get_plot_data(test_nominal_loader)
    #    test_target_xi = get_plot_data(test_target_loader)
    #
    #    # Plotting
    #    carl_names = ["Target"]
    #    r_hat_list = [np.ones(np.shape(test_nominal_xi[1])[0])]
    #    utils.plotting.plot_distributions(test_nominal_xi[0], test_target_xi[0],
    #                                      test_nominal_xi[1], r_hat_list, test_target_xi[1],
    #                                      carl_names=carl_names,
    #                                      feature_name="$x_{}$".format(i), 
    #                                      alternate_name="Toy Camel",
    #                                      percentile_cuts=(0.1,99.99),
    #                                      nominal_mask=np.isfinite, 
    #                                      alternate_mask=np.isfinite, 
    #                                      carl_mask=np.isfinite, 
    #                                      logscale=False,
    #                                      typical_ratio = False,
    #                                      #ref_name = "Optimal",
    #                                      saveAs=f'x_{i}.pdf',
    #                                      nbins=100,
    #                                      global_name = 'Toy Model',
    #                                      **kwargs  )

    # Radial direction
    #test_nominal_loader = DataLoader(utils.preprocessing.CombinedDataset(test_base_dataset), 
    #                                 batch_size=batch_size, 
    #                                 shuffle=False)
    #test_target_loader  = DataLoader(utils.preprocessing.CombinedDataset(test_target_dataset), 
    #                                 batch_size=batch_size, 
    #                                 shuffle=False)
    
    test_nominal_loader.collate_fn = lambda batch: get_r(batch)
    test_target_loader.collate_fn = lambda batch: get_r(batch)
        
    test_nominal_r = get_plot_data(test_nominal_loader)
    test_target_r = get_plot_data(test_target_loader)
    # Plotting
    carl_names = ["Target"]
    r_hat_list = [np.ones(np.shape(test_nominal_r[1])[0])]
    hist_x0, hist_x1, binning = utils.plotting.plot_distributions(test_nominal_r[0], test_target_r[0],#+3,
                                                                  test_nominal_r[1], r_hat_list, test_target_r[1],
                                                                  carl_names=carl_names,
                                                                  feature_name="$r$", 
                                                                  alternate_name="Toy Camel",
                                                                  percentile_cuts=(0.1,99.99),
                                                                  nominal_mask=np.isfinite, 
                                                                  alternate_mask=np.isfinite, 
                                                                  carl_mask=np.isfinite, 
                                                                  logscale=False,
                                                                  typical_ratio = False,
                                                                  saveAs='r.pdf',
                                                                  nbins = n,
                                                                  global_name = 'Toy Model',
                                                                  **kwargs )
    
        


    # Calculate the euclidean cost matrix
    radial_positions = binning[:-1]
    radial_positions += (radial_positions[1] - radial_positions[0]) / 2
    #M = real_euc_costs( radial_positions )
    M = euc_costs(n-1,n-1)

    # Define empty transport matrix - P
    P = []
    
    # Determine number of q-parameter tests
    nq = len(q)
    # Determine the number of regularisation tests
    nl = len(l)
    
    #hist_x0 = np.abs(hist_x0)
    #hist_x0 = -1*hist_x0
    #hist_x0 = hist_x1
    hist_x1 = np.abs(hist_x1)

    #mu = 0.35
    #hist_x1 = ( ((1-mu)*hist_x0) + (mu*hist_x1))

    mu = 2.0
    #hist_x1 = ( ((1-mu)*hist_x0) + (mu*hist_x1))
    #hist_x0 = ( ((mu)*hist_x0) + ((1-mu)*hist_x1))


    # Tests
    #hist_x0 -= poisson.pmf(range(n-1), 5)
    hist_x1 -= mu*poisson.pmf(range(n-1), 30)


    hist_x0 /= np.sum(hist_x0)#/100
    hist_x1 /= np.sum(hist_x1)#/100
    print(f' Int a = {np.sum(hist_x0)}')
    print(f' Int b = {np.sum(hist_x1)}')

    #hist_x1 = -1*hist_x1

    #hist_x0 = np.abs(hist_x0)
    #hist_x0 = -1*hist_x0
    #hist_x0 = hist_x1
    #hist_x1 = np.abs(hist_x1)
    #hist_x0 = hist_x0**2
    #hist_x1 = hist_x1**2

    # Produce the transportation matrices
    for j in range(nq):
        for i in range(nl):
            print(q[j],l[i])
            P_tmp = TROT(q[j],M,hist_x1,hist_x0,l[i],1E-7)
            #P_tmp = np.zeros( np.shape(M) )
            P.append(P_tmp)
    
    
    # Now generate a figure of the various marginals and transport matrices
    fig = plt.figure(figsize=(12, 12))
    
    # Generate the outter grid of the plots
    outer_grid = gridspec.GridSpec(2, 2, width_ratios=[1,5], height_ratios=[1,5])
    outer_grid.update(wspace=0.01, hspace=0.01)
    # gridspec inside gridspec
    outer_joint = gridspec.GridSpecFromSubplotSpec(nq,nl, subplot_spec=outer_grid[1,1],wspace=0.02, hspace=0.02) #0.5
    outer_row_marg = gridspec.GridSpecFromSubplotSpec(nq,1, subplot_spec=outer_grid[1,0],wspace=0.02, hspace=0.02) #0.5
    outer_col_marg = gridspec.GridSpecFromSubplotSpec(1,nl, subplot_spec=outer_grid[0,1],wspace=0.02, hspace=0.02) #0.5
    
    #  -> Transport matrices
    EMD = []
    COST = []
    for b in range(nl):
        for a in range (nq):
            # Subplot axis
            ax = plt.Subplot(fig, outer_joint[a,b])

            # Heatmap plotting
            heatmap = ax.imshow(P[nl*a + b], origin='upper', interpolation = None, aspect = 'auto', cmap = 'plasma', norm='log')
            # Colour bar
            #fig.colorbar(heatmap, ax=ax, location='right')


            EMD.append(P[nl*a + b]*M)#euc_costs(n-1,n-1))
            COST.append(M)
            #rect = Rectangle((0, 0), n-1, n-1, fc='none', ec='black')     
            #rect.set_width(0.8)
            #rect.set_bounds(0,0,n-1,n-1)
            #ax.add_patch(rect)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
            #ax.set_facecolor('white')
            #print(f'W_{q[a],l[b]} = {np.sum(P[nl*a + b]*euc_costs(n-1,n-1))}')
            print(f'W_{q[a],l[b]} = {np.sqrt(np.sum(P[nl*a + b]*M))}')
            
    for i in range(nq):
        ax_row = plt.Subplot(fig,outer_row_marg[i], sharey = ax)
        #ax_row.plot(1-hist_x1, np.arange(0, np.shape(radial_positions)[0]), color='orange' )#radial_positions)
        ax_row.plot(hist_x1, np.arange(0, np.shape(radial_positions)[0]), color='orange' )#radial_positions)
        ax_row.plot(np.zeros(np.shape(radial_positions)[0]), np.arange(0, np.shape(radial_positions)[0]), color='grey', linestyle='dashed' )
        fig.add_subplot(ax_row)
        
        ax_row.axes.get_xaxis().set_visible(False)
        ax_row.axes.get_yaxis().set_visible(False)
        bottom, height = .25, .5
        top = bottom + height
        ax_row.text(-0.05, 0.5*(bottom+top), 'q = %.2f' % q[i], horizontalalignment='right', verticalalignment='center', rotation='vertical',transform=ax_row.transAxes, fontsize='medium')
        #ax_row.set_facecolor('white')
    
    for j in range(nl):
        ax_col = plt.Subplot(fig,outer_col_marg[j], sharex = ax)
        ax_col.plot(np.arange(0, np.shape(radial_positions)[0]), hist_x0)
        ax_col.plot(np.arange(0, np.shape(radial_positions)[0]), np.zeros(np.shape(radial_positions)[0]),
                    color='grey', linestyle='dashed' )
        fig.add_subplot(ax_col)    
        bottom, height = .25, .5
        ax_col.axes.get_xaxis().set_visible(False)
        ax_col.axes.get_yaxis().set_visible(False)
        ax_col.set_title(r'$\lambda$'+' = {0}'.format(l[j]),fontsize='medium')
        #ax_col.set_facecolor('white')
        
    #fig.show()
    plt.savefig('TROT_image-neg.pdf', format='pdf')

    Plot(q, l, nq, nl, n,  
         EMD, 
         hist_x0, hist_x1, radial_positions, 
         file_name = 'TROT_EMD-neg')

    Plot(q, l, nq, nl, n,  
         COST, 
         hist_x0, hist_x1, radial_positions, 
         file_name = 'TROT_COST-neg')
    
    '''
    n = 50
    q = [2.0, 1.0, 0.5] #np.arange(0.5,2.5,0.5)
    l = [50, 25, 10, 1]
    mu1 = [10,30]
    mu2 = [5, 20, 35]
    t1 = [0.5  , 0.5 ]
    t2 = [0.2 , 0.8 , 0.2]


    x = range(n)
    
    r_tmp = []    
    for mode in mu1:
        r_tmp.append(poisson.pmf(x,mode))
        
    c_tmp = []    
    for mode in mu2:
        #if mode == 5:
        #    c_tmp.append(-1*poisson.pmf(x,mode))
        #else:
        c_tmp.append(poisson.pmf(x,mode))
        
    r = np.dot(t1,r_tmp)
    r = r/r.sum()
    
    c = np.dot(t2,c_tmp)
    c = c/c.sum()
    
    
    M = euc_costs(n,n)
    
    P = []
    
    nq = len(q)
    nl = len(l)
    
        
    for j in range(nq):
        for i in range(nl):
            P_tmp = TROT(q[j],M,r,c,l[i],1E-2)
            P.append(P_tmp)
    
    
    fig = plt.figure(figsize=(8, 8))
    
    outer_grid = gridspec.GridSpec(2, 2, width_ratios=[1,5], height_ratios=[1,5])
    outer_grid.update(wspace=0.01, hspace=0.01)
    # gridspec inside gridspec
    outer_joint = gridspec.GridSpecFromSubplotSpec(nq,nl, subplot_spec=outer_grid[1,1],wspace=0.02, hspace=0.02)
    outer_row_marg = gridspec.GridSpecFromSubplotSpec(nq,1, subplot_spec=outer_grid[1,0],wspace=0.02, hspace=0.02)
    outer_col_marg = gridspec.GridSpecFromSubplotSpec(1,nl, subplot_spec=outer_grid[0,1],wspace=0.02, hspace=0.02)
    
    
    for b in range(nl):
        for a in range (nq):
            ax = plt.Subplot(fig, outer_joint[a,b])
            ax.imshow(P[nl*a + b], origin='upper', interpolation = None, aspect = 'auto', cmap = 'Greys')
            rect = Rectangle((0, 0), n-1, n-1, fc='none', ec='black')     
            rect.set_width(0.8)
            rect.set_bounds(0,0,n-1,n-1)
            ax.add_patch(rect)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
            ax.set_facecolor('white')
            
    for i in range(nq):
        ax_row = plt.Subplot(fig,outer_row_marg[i], sharey = ax)
        ax_row.plot(1-r, x)
        fig.add_subplot(ax_row)
        
        ax_row.axes.get_xaxis().set_visible(False)
        ax_row.axes.get_yaxis().set_visible(False)
        bottom, height = .25, .5
        top = bottom + height
        ax_row.text(-0.05, 0.5*(bottom+top), 'q = %.2f' % q[i], horizontalalignment='right', verticalalignment='center', rotation='vertical',transform=ax_row.transAxes, fontsize='medium')
        
        ax_row.set_facecolor('white')
    
    for j in range(nl):
        ax_col = plt.Subplot(fig,outer_col_marg[j], sharex = ax)
        ax_col.plot(x,c)
        fig.add_subplot(ax_col)    
        bottom, height = .25, .5
        ax_col.axes.get_xaxis().set_visible(False)
        ax_col.axes.get_yaxis().set_visible(False)
        ax_col.set_title(r'$\lambda$'+' = {0}'.format(l[j]),fontsize='medium')
        ax_col.set_facecolor('white')
        
    #fig.show()
    plt.savefig('TROT_image.pdf', format='pdf')
    '''

# Main call
if __name__ == "__main__":
    main()
