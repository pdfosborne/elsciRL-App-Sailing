from datetime import datetime
import pandas as pd
# ====== elsciRL IMPORTS =========================================
# ------ Train/Test Function Imports ----------------------------
from elsciRL import STANDARD_RL
from elsciRL import elsciRL_SEARCH
from elsciRL import elsciRL_OPTIMIZE
# ------ Config Import ------------------------------------------
# Meta parameters
from elsciRL.config import TestingSetupConfig
# Local parameters
from elsciRL.config_local import ConfigSetup
# ====== LOCAL IMPORTS ==========================================
# ------ Local Environment --------------------------------------
from environment.engine import Engine
# ------ ADAPTERS -----------------------------------------------
from adapters.default import DefaultAdapter
from adapters.language import LanguageAdapter
ADAPTERS = {"Default": DefaultAdapter, "Language": LanguageAdapter}

def main():
    # ------ Load Configs -----------------------------------------
    # Meta parameters
    ExperimentConfig = TestingSetupConfig("./configs/config.json").state_configs
    # Local Parameters
    ProblemConfig = ConfigSetup("./configs/config_local.json").state_configs

    # Specify save dir
    time = datetime.now().strftime("%d-%m-%Y_%H-%M")
    save_dir = './output/'+str('test')+'_'+time 

    # elsciRL Instruction Following
    num_plans = 50
    num_explor_epi = 100000
    sim_threshold = 0.9

    observed_states = None
    instruction_results = None
    
    search_agent = elsciRL_SEARCH(Config=ExperimentConfig, LocalConfig=ProblemConfig, 
                        Engine=Engine, Adapters=ADAPTERS,
                        save_dir = save_dir+'/Reinforced_Instr_Experiment',
                        num_plans = num_plans, number_exploration_episodes=num_explor_epi, sim_threshold=sim_threshold,
                        feedback_increment = 0.1, feedback_repeats=1,
                        observed_states=observed_states, instruction_results=instruction_results)

    # Don't provide any instruction information, will be defined by command line input
    results = search_agent.search(action_cap=100, re_search_override=False, simulated_instr_goal=None)

    # Store info for next plan -> assumes we wont see the same instruction twice in one plan
    observed_states = results[0]
    instruction_results = results[1]
    # Take Instruction path now defined with reinforced+unsupervised sub-goal locations and train to these
    # Init experiment setup with sub-goal defined
    reinforced_experiment = elsciRL_OPTIMIZE(Config=ExperimentConfig, LocalConfig=ProblemConfig, 
                    Engine=Engine, Adapters=ADAPTERS,
                    save_dir=save_dir+'/Reinforced_Instr_Experiment', show_figures = 'No', window_size=0.1,
                    instruction_path=instruction_results, predicted_path=None, instruction_episode_ratio=0.05,
                    instruction_chain=True, instruction_chain_how='exact' )
    reinforced_experiment.train()
    reinforced_experiment.test()
    
    # --------------------------------------------------------------------
    # Flat Baselines
    flat = STANDARD_RL(Config=ExperimentConfig, LocalConfig=ProblemConfig, 
                Engine=Engine, Adapters=ADAPTERS,
                save_dir=save_dir, show_figures = 'No', window_size=0.1)
    flat.train()  
    flat.test()
    # --------------------------------------------------------------------


if __name__=='__main__':
    main()