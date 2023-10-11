from dosmodel import *

directory = os.getcwd()
csv_files = get_csv_files(directory)

# csv_files = [list of csv files]

for file in csv_files:
    head, tail = ntpath.split(file)
    print(tail.replace('.csv', ''), end = ' -->', flush  = True)
    dos_model = DOSModel(file)
    for fx in fx_list:
        print(str(fx.__name__), end = ', ', flush = True)
        dos_model.do_fitting(fx)
        dos_model.plot_exp_pred()
        dos_model.plot_2d()
        dos_model.plot_2d_local()
        dos_model.plot_3d()
        dos_model.plot_contour()
        dos_model.calc_deriv() 
        dos_model.calc_deriv_smooth() 
        dos_model.plot_contour_norm()
        dos_model.plot_contour_norm_local()
        
