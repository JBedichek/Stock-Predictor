# Stock-Predictor
(The contrastive_pretraining file is only used for the gauss_normalize function, ignore the rest it wasn't effective)

training.py - Training loops and dataset generation
stock.py - Data scraping
utils.py - Misc. functions
inference.py - Running trading simulations and generating inference datasets

MAIN FUNCTIONALITY SUMMARIES:

training.py:
  GenerateDataDict - Creates a dictionary of price values and day of the week to feed to QTrainingData

  QTrainingData - (Name is old from when I tried RL) The dataset generation class, taking information from GenerateDataDict and other 
  information generated from stock.py to create datasets either for training or inference (Also has functionality for greedy layer wise 
  pretraining, i.e layer_args, ignore that)

  Train_Dist_Direct_Predictor - main training loop, takes a list of dataset paths and runs training from a pre-generated or loaded model

  Train_Dist_Direct_Predictor_SAM - training loop with Sharpness Aware Minimization

stock.py:
  stock_info - data scraping class, takes a dictionary of stocks ({ticker symbol:company_name}) and uses the yfinance library to scrape stock 
  info and save them as pickle files to the cwd

models.py:
  t_Dist_Pred - Main model class, mean pooling transformer encoder with stochastic depth

inference.py:
  generate_current_day_dataset - generates an inference dataset for the current day

  do_current_day_prediction - runs a trading sim based on hard-coded datasets (dataset_pth)

  stock_inference - main trading sim class, for running trading simulations (run_trading_sim)
    
    get_top_q_buy - returns the n highest prediction companies for a given day



To run a simulation:
Make sure you have the relevant datasets and model in the cwd, then confiture your trading parameters and run the file:
  trading parameters (variables in the main function near the bottom of the file):
    c_prune - the proportion of companies to run inference on (~4000 total companies)
    low - if true, sells companies short instead of buying them
    entropy - whether to pick the one company in the top n with the lowest entropy (highest confidence) predition
    top_n - number of companies to trade each day (lower is higher variance, about 10 I've found to be a good balance)
    e_high - (experimental, not reccomended) pick the one company with the highest entropy
    generate - if true, generate an inference dataset
    ignore_thr - set to true, was an experimental thing where it would only trade on days where it gave unusually high predictions
    use_meta_model - whether to use a meta model (a model which is trained to predict how wrong the main model's prediction is, set to False 
                     for now)
    mm_prop - the proportion of companies to keep when using a meta model (only keep the highest confidence predicitons)
    track_wandb - whether to log the results of the sim using WandB
