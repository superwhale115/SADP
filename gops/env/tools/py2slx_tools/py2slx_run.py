# Parameter Description:
"""
   'log_policy_dir_list' is the trained policy loading path
   'trained_policy_iteration_list' is the trained policy corresponding to the number of iteration steps
   'export_controller_name' is the name of the export controller you want
   'save_path' is the absolute save path of the export controller,preferably in the same directory as the simulink project files
"""

from py2slx import Py2slxRuner

runer = Py2slxRuner(
    log_policy_dir_list=[r"D:\2_Genjin\THU\Code\gops\results\DDPG\221028-203632"],
    trained_policy_iteration_list=['250000'],
    export_controller_name=['NN_controller_DDPG'],
    save_path=[r'C:\Users\Genjin Xie\Desktop\GOPS_test\LQ_model']
    )

runer.simulink()
