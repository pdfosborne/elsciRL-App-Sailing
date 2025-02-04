import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

@staticmethod
def vel(theta, theta_0=0, theta_dead=np.pi / 12):
    return 1 - np.exp(-(theta - theta_0) ** 2 / theta_dead)

@staticmethod
def rew(theta, theta_0=0, theta_dead=np.pi / 12):
    return vel(theta, theta_0, theta_dead) * np.cos(theta)


class Analysis:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        
    def trace_plot(self):
        """Define experiment directories to run and override names for
        improved plot titles."""
        path = self.save_dir 
        path_folders = os.listdir(path)
        plot_list = {}
        for n,folder in enumerate(path_folders):
            figure = plt.figure()
            ax = figure.add_subplot(1, 1, 1)
            if os.path.isdir(path+'/'+folder):
                exp_path = path + '/' + folder
                exp_path_folders = os.listdir(exp_path)

                count_check = 0
                previous_agent = ''
                policy_output_dict = {}
                for result_folders in exp_path_folders:
                    if os.path.isdir(exp_path+'/'+result_folders):
                        if 'training' in result_folders:
                            testing_results_path = exp_path + '/' + result_folders
                            path_csv=glob.glob(testing_results_path+"/*.csv")
                            results = pd.read_csv(path_csv[0])
                            agent = results['agent'].iloc[0].split('__')[0]
                            # Update output dict with policy list
                            if agent != previous_agent:
                                policy_output_dict[previous_agent] = policy_list
                                policy_list = []
                                
                            policy = results['action_history'].mode()[0]
                            policy_fix = policy.split(',')
                            policy_fix = [int(i.replace('[1','1').replace('[0','0').replace('1]','1').replace('0]','0')) for i in policy_fix]
                            policy_list.append(policy_fix)
                            count_check += 1
                            previous_agent = agent
                # Add final policy list to output dict
                policy_output_dict[previous_agent] = policy_list
                for agent in policy_output_dict.keys():
                    policy_list = policy_output_dict[agent]
                    #print(count_check)
                    # Re-applies actions made by agent to observe path
                    if 'instr' in folder.lower():
                        exp_title = path.split('/')[-1] + ' - ' + agent
                    else:
                        exp_title = folder + ' - ' + agent
                    
                    ax.scatter(0,0,marker='x', color='b')
                    training_policies = policy_list
                    for action_list in training_policies:
                        x = 0
                        y = 0
                        angle = 0
                        x_list=[]
                        y_list=[]
                        for action in action_list:
                            a = [-0.1,0.1][action]
                            
                            #print(x,"|", y, "|", angle)
                            x += np.round((vel(angle + a) * np.sin(angle + a)),4) # Round x to 2dp
                            y += np.round((vel(angle + a) * np.cos(angle + a)),4) # Round y to 2dp
                            angle=np.around(angle+a,1)
                            # if (x > 3)&(x<5):
                            #     print("---")
                            #     print("{:0.4f}".format(x),"|", "{:0.4f}".format(y), "|", "{:0.1f}".format(angle))
                            x_list.append(x)
                            y_list.append(y)

                        if np.abs(x_list[-1])>=10:
                            ax.plot(x_list,y_list,'r',alpha=0.75)
                        elif np.abs(y_list[-1])>=24:
                            ax.plot(x_list,y_list,'g',alpha=0.75)
                        elif np.abs(y_list[-1])<0:
                            ax.plot(x_list,y_list,'r',alpha=0.75)
                        else:
                            ax.plot(x_list,y_list,'k',alpha=0.75)
                            
                        ax.scatter(x_list[-1],y_list[-1],marker='x', color='r')
                        ax.plot([10,10],[0,25],'r')
                        ax.plot([-10,-10],[0,25],'r')
                        figure.suptitle(exp_title + "\n Sailboat Path for each Trained Agent's Output Policy")
                        ax.set_xlabel("Horizontal Position (x)")
                        ax.set_ylabel("Vertical Position (y)")
                    
                    #save_path = os.path.join(path, folder, 'trace_plot.png')

                    # plt.savefig(save_path)
                    plt.show()
                    plt.close()

            plot_list['plot'+str(n)] = figure

        return plot_list

                
