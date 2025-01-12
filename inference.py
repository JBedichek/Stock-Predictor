import torch
import torch.nn as nn
import datetime
from utils import *
from models import *
from training import QTrainingData, GenerateDataDict
from tqdm import tqdm
import os
import random 

def generate_current_day_dataset(i_keep):
    from stock import generate_day_data
    #generate_day_data()
    #bulk_prices_pth = 'unnorm_price_series_prediction_-5y-'+ f'{datetime.date.today()}'+'__.pickle'
    bulk_prices_pth = 'unnorm_price_series_prediction_-5y-'+ f'2024-12-15'+'__.pickle'


    summaries_pth = 'pred_summary_embs.pickle'
    sectors_pth = 'prediction_sector_dict'
    #data_dict_pth = f'pred_DataDict_{datetime.date.today()}'
    data_dict_pth = f'pred_DataDict_{datetime.date(2024,12,15)}'
    # Generate Data Dict for inference Dataset
    #data = GenerateDataDict(bulk_prices_pth, summaries_pth, 600, 
    #                save=True, save_name=data_dict_pth)

    # Create Dataset using DataDict
    dataset = QTrainingData(summaries_pth, bulk_prices_pth, 600, full=False, load=1, data_dict_pth=data_dict_pth,
                  inference=(datetime.date(2024, 11, 12), datetime.date(2024, 12,5)), sectors_pth=sectors_pth,
                  inf_company_keep_rate=i_keep)
    dataset_pth = f'{datetime.date.today()}_prediction_dataset'
    #save_pickle(dataset, dataset_pth)
    torch.save(dataset.inference_data, dataset_pth+'.pt')


def do_current_day_prediction(model_pths, n, top_n, low, entropy=0, steps=2, e_high=0, debug=0, generate=1, 
                              c_prune=1, ignore_thr=1, use_meta_model=True,mm_prop=0.25,track_wandb=0,i_keep=1):
    double=1
    if generate:
        generate_current_day_dataset(i_keep)
    dataset_pth = f'{datetime.date(2024,12,29)}_prediction_dataset'
    inf = stock_inference(model_pths,None,None,None,start_date=None,end_date=None,load_dataset=True,n=n,
                        debug=debug,custom_dataset=dataset_pth,meta_model=use_meta_model)
    #inf.run_random_trading_sims(datetime.date(2024,11,25),13,200)
    if not double:
        inf.run_trading_sim(top_n, datetime.date(2024,10,25), 40, low=low, entropy=entropy, steps=steps, 
                e_high=e_high, company_prune=c_prune, ignore_thr=ignore_thr, use_meta_model=use_meta_model,
                score_prop=mm_prop,track_wandb=track_wandb)
    else:
        try:
            inf.run_trading_sim(top_n, datetime.date(2024,9,25), 11, low=low, entropy=entropy, steps=steps, 
                e_high=e_high, company_prune=c_prune, ignore_thr=ignore_thr, use_meta_model=use_meta_model,
                score_prop=mm_prop,track_wandb=track_wandb)
        except Exception as e:
            pass
        money = inf.cum_money
        bp = inf.bool_profit
        dataset_pth = f'{datetime.date(2024,12,29)}_prediction_dataset_1'
        inf = stock_inference(model_pths,None,None,None,start_date=None,end_date=None,load_dataset=True,n=n,
                        debug=debug,custom_dataset=dataset_pth,meta_model=use_meta_model)
        inf.cum_money = money
        inf.bool_profit = bp
        try:
            inf.run_trading_sim(top_n, datetime.date(2024,10,10), 40, low=low, entropy=entropy, steps=steps, 
                e_high=e_high, company_prune=c_prune, ignore_thr=ignore_thr, use_meta_model=use_meta_model,
                score_prop=mm_prop,track_wandb=track_wandb)
        except Exception as e:
            pass
        money = inf.cum_money
        bp = inf.bool_profit
        dataset_pth = f'{datetime.date(2024,12,31)}_prediction_dataset_1'
        inf = stock_inference(model_pths,None,None,None,start_date=None,end_date=None,load_dataset=True,n=n,
                        debug=debug,custom_dataset=dataset_pth,meta_model=use_meta_model)
        inf.cum_money = money
        inf.bool_profit = bp
        try:
            inf.run_trading_sim(top_n, datetime.date(2024,10,25), 40, low=low, entropy=entropy, steps=steps, 
                e_high=e_high, company_prune=c_prune, ignore_thr=ignore_thr, use_meta_model=use_meta_model,
                score_prop=mm_prop,track_wandb=track_wandb)
        except Exception as e:
            pass
        money = inf.cum_money
        bp = inf.bool_profit
        dataset_pth = f'{datetime.date(2025,1,4)}_prediction_dataset'
        inf = stock_inference(model_pths,None,None,None,start_date=None,end_date=None,load_dataset=True,n=n,
                        debug=debug,custom_dataset=dataset_pth,meta_model=use_meta_model)
        inf.cum_money = money
        inf.bool_profit = bp
        inf.run_trading_sim(top_n, datetime.date(2024,11,15), 40, low=low, entropy=entropy, steps=steps, 
                e_high=e_high, company_prune=c_prune, ignore_thr=ignore_thr, use_meta_model=use_meta_model,
                score_prop=mm_prop,track_wandb=track_wandb)

class stock_inference:
    def __init__(self, model_pths, s_pth, p_pth, sectors_pth, device='cuda', start_date=(2024, 5, 1), 
                 end_date=(2024, 5, 1), n=3, seq_len=600, num_bins=320, load_dataset=False, 
                 data_dict_pth='DataDict_2024-06-26', debug=False, custom_dataset=False, meta_model=None):
        # Data dict -> date : company : data(tensor)
        self.num_bins = num_bins

        if meta_model:
            self.meta_model = torch.load('MetaModel')
            print('Loaded Meta Model...')
            self.use_meta_model = True
        else:
            self.use_meta_model = False
        # This is unnecessary when doing a single day prediction
        if (start_date and end_date) is not None:
            self.start_date = datetime.date(start_date[0], start_date[1], start_date[2])
            self.end_date = datetime.date(end_date[0], end_date[1], end_date[2])
        print('Loading Model(s)')
        self.model = [torch.load(model_pth).eval().to(device) for model_pth in model_pths]
        self.model = [torch.compile(item,mode='max-autotune') for item in self.model]
        if load_dataset:
            print('Loading Dataset (may take a minute)...')
            if custom_dataset:
                #self.dataset = torch.load(f'{datetime.date.today()}_prediction_dataset'+".pt")
                self.dataset = torch.load(custom_dataset+".pt")
                #self.dataset = pic_load(custom_dataset)
                #torch.save(self.dataset, 'InfDataset.pt')
            else:
                self.dataset = torch.load('InfDataset.pt')
                #self.dataset = pic_load('InferenceDataset')
            print('Loaded Dataset...')
        else:
            
            self.dataset = QTrainingData(s_pth, p_pth, seq_len, n=n, full=False, load=1, data_dict_pth=data_dict_pth,
                                        inference=(self.start_date, self.end_date), sectors_pth=sectors_pth,
                                    )
            save_pickle(self.dataset, 'InferenceDataset')
        self.iterable_dates = list(self.dataset.keys())
        print('Dataset Range: ',self.iterable_dates[0],self.iterable_dates[-1])
        #print(len(self.iterable_dates))
        self.bin_edges = torch.load('Bin_Edges_300')
        #print('Num companies: ', len(self.companies))
        self.i_dataset = self.dataset
        a = list(self.i_dataset.keys())
        self.companies = list(self.i_dataset[a[0]].keys())
        self.n = n
        self.cum_money = 80000
        self.debug = debug
        self.rank_profits = []
        self.bool_profit = []
        


    def get_company_date_price(self, company, date):
        day_data = self.i_dataset[date][company][2].item()
        #print(day_data)
        return day_data
    
    def get_expected_q(self, pred, device='cuda'):
        '''
        Takes a vector of logits, corresponding to probability bins, and outputs a mean
        price prediction.
        '''
        #softmax = nn.Softmax(dim=0)
        device = pred.device
        #reward_values = torch.linspace(-0.2, 0.2, self.num_bins).repeat(2, 1).to(device)
        #reward_values = torch.linspace(-0.2, 0.2, self.num_bins, device=device)
        reward_values = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        reward_values = reward_values.to(device)
        #expected_reward = torch.sum(reward_values*softmax(pred), dim=1)
        expected_reward = torch.sum((reward_values*pred), dim=0)
        #r_val = torch.max(expected_reward).item()
        return expected_reward
    
    @torch.compile
    def get_top_q_buy(self, top_n, date, show=True, low=False, entropy=True, steps=1, e_high=True,
            buy_sell=False,correct_for_date=True,use_jac_norm=False, score_prop=0.1, use_meta_model=False):
        '''
        Iterates through all companies in the dataset on a given date, gets
        the model predictions, and selects the top_n best companies to buy based
        on those predictions.
        '''
        self.use_meta_model = use_meta_model
        print(date)
        num_c = int(self.c_prune*len(self.companies))
        effective_companies = random.sample(self.companies, num_c)
        ind = 0
        with torch.no_grad():
            pred = {}
            best_pred = {}
            entropies = {}
            c2_jac_norms = {}
            jac_norms_2c = {}
            dists = {}
            if use_meta_model:
                c2_mm_scores = {}
                mm_scores_2c = {}
            for company in tqdm(effective_companies):
                if self.debug:
                    data = self.i_dataset[date][company]
                    if data is None:
                        pass
                    else:
                        preds = []
                        data, summary, price = data[0].unsqueeze(0), data[1].unsqueeze(0), data[2]
                        data = set_nan_inf(data)
                        #for model in self.models:
                        if use_meta_model:
                            price_pred, t_act = self.model[0].forward_with_t_act(data, summary)
                            mm_score = self.meta_model(t_act, price_pred)
                            c2_mm_scores[company] = mm_score
                            mm_scores_2c[mm_score] = company
                            # TODO: Use scores to shorten company list


                        price_pred = noise_averaged_inference(self.model, data, summary, 
                                                steps)
                        preds.append(price_pred)
                        
                        n = price_pred.size(0)
                        prices = []
                        _entropies = []
                        jacobian_norm = 0
                        if use_jac_norm:
                            for model in self.model:
                                jacobian_norm += get_jacobian_norm(model,data,summary)
                        for i in range(4):
                            prices.append(self.get_expected_q(price_pred[:,i], self.num_bins))
                            _entropies.append(get_distribution_entropy(price_pred[:,i]))
                        if correct_for_date:
                            price_pred = prices[self.n-2]/prices[self.n-3]
                        else:
                            price_pred = prices[self.n-2]
                        pred[company] = price_pred[i]
                        entropies[company] = _entropies[self.n-2]
                        best_pred[price_pred] = company
                        if use_jac_norm:
                            jac_norms_2c[jacobian_norm] = company
                            c2_jac_norms[company] = jacobian_norm
                else:
                    try:
                        data = self.i_dataset[date][company]
                        if data is None:
                            pass
                        else:
                            data, summary, price = data[0].unsqueeze(0), data[1].unsqueeze(0), data[2]
                            data = set_nan_inf(data)
                            #print(data.shape)
                            if use_meta_model:
                                price_pred, t_act = self.model[0].forward_with_t_act(data.to('cuda'), summary.to('cuda'))
                                price_pred = price_pred.squeeze(0)
                                
                                #print(price_pred.shape, t_act.shape)
                                mm_score = self.meta_model(t_act.unsqueeze(0), price_pred.unsqueeze(0))
                                mm_score = mm_score[0]
                                c2_mm_scores[company] = mm_score
                                mm_scores_2c[mm_score] = company
                                
                            else:
                                price_pred = noise_averaged_inference(self.model, data, summary, 
                                                steps)
                            dists[company] = price_pred[:,0]
                            n = price_pred.size(0)
                            prices = []
                            _entropies = []
                            jacobian_norm = 0
                            if use_jac_norm:
                                for model in self.model:
                                    jacobian_norm += get_jacobian_norm(model,data,summary)
                                    print(jacobian_norm)
                            for i in range(4):
                                prices.append(self.get_expected_q(price_pred[:,i], self.num_bins))
                                _entropies.append(get_distribution_entropy(price_pred[:,i]))
                            if correct_for_date:
                                
                                price_pred = prices[self.n-2]-prices[self.n-3]
                            else:
                                price_pred = prices[self.n-2]
                            #if use_meta_model:
                            #    price_pred=price_pred+mm_score
                            pred[company] = price_pred
                            entropies[company] = _entropies[self.n-2]
                            best_pred[price_pred] = company
                            if use_jac_norm:
                                jac_norms_2c[jacobian_norm] = company
                                c2_jac_norms[company] = jacobian_norm
                    except Exception as e:
                    #    print(e)
                        pass
        if use_meta_model:
            sorted_mm_scores = sorted(list(c2_mm_scores.values()))
            useable_mm_scores = sorted_mm_scores[:int(len(sorted_mm_scores)*score_prop)]
            companies = [mm_scores_2c[item] for item in useable_mm_scores]
            best_pred = {key:value for key, value in best_pred.items() if value in companies}

        if use_jac_norm:
            sorted_jac_norms = sorted(list(jac_norms_2c.keys()))
            print(len(sorted_jac_norms))
            usable_jac_norms = sorted_jac_norms[:int(len(sorted_jac_norms)*score_prop)]
            print(len(usable_jac_norms))
            companies = [jac_norms_2c[item] for item in usable_jac_norms]
            print(len(companies))
            best_pred = {key:value for key, value in best_pred.items() if value in companies}
            print(len(list(best_pred.keys())))
        b_p = sorted(list(best_pred.keys()))
        #print(b_p[-10:], b_p[:10])


        if buy_sell:
            top_n = 1
            bb_p = b_p[:top_n]
            sb_p = b_p[-top_n:]
            if abs(sb_p[0]) > sb_p[0]:
                b_p = sb_p
                low=True
            else:
                b_p = bb_p
        else:
            if low:
                b_p = b_p[:top_n]
            else:
                b_p = b_p[-top_n:]
            
        n_companies = [best_pred[p] for p in b_p]
        print(n_companies)
        if entropy:
            if e_high:
                entropy_values = [entropies[company] for company in n_companies]
                min_entropy_index = entropy_values.index(max(entropy_values))
                n_companies = [n_companies[min_entropy_index]]
            else:
                entropy_values = [entropies[company] for company in n_companies]
                min_entropy_index = entropy_values.index(min(entropy_values))
                n_companies = [n_companies[min_entropy_index]]
                p = pred[n_companies[0]]
        if show:
            #print('TOP PROFIT PREDICTIONS')
            if entropy:
                #print(n_companies[0], ' ', f'{100*pred[n_companies[0]]:.5f}')
                print(n_companies)
            else:
                for p in b_p:
                    c = best_pred[p]
                    #print(p)
                    #print(dists[c])
                    print(c, ' ', f'{100*p:.10f}%')
        if n_companies is not None:

            return n_companies, 100*p
        else: 
            print(best_pred)
            raise ValueError
        
    def get_date_profit(self, date, companies, low=False, offset=True,counter=1):
        if len(companies) == 0:
            print('No Companies!')
            return None
        #ind = self.dataset.iterable_dates.index(date) 
        ind = self.iterable_dates.index(date)+1 
        #buy_date = self.dataset.iterable_dates[self.dataset.iterable_dates.index(date)+1] 
        buy_date = self.iterable_dates[self.iterable_dates.index(date)+1] 
        do = 1
        try:
            buy_prices = [self.get_company_date_price(c, buy_date) for c in companies]
            #sell_date = self.dataset.iterable_dates[ind+(self.n-1)]
            sell_date = self.iterable_dates[ind+(self.n-1)+1]
            sell_prices = [self.get_company_date_price(c, sell_date) for c in companies]
        except Exception as e:
            print('Not found: ', e, 'Assuming zero profit')
            buy_prices = [1]
            sell_prices = [1]
            #sell_date = self.dataset.iterable_dates[ind+(self.n-1)]
            sell_date = self.iterable_dates[ind+(self.n-1)]
            do = 0
        profit = []
        print(f'Buy: {buy_date}, Sell: {sell_date}')
        for i in range(len(buy_prices)):
            profit.append(sell_prices[i]/buy_prices[i])
        #print(profit)
        for i in range(len(profit)):
            self.rank_profits_acc[i] += profit[i]
        if do:
            print('Company-wise profits')

            for i in range(len(companies)):
                print(f'{companies[i]}: {profit[i]:.4f} buy: {buy_prices[i]:4f} sell: {sell_prices[i]:4f} pos profit avg: {self.rank_profits_acc[i]/(counter+1)}')
            #print(profit)
        profit = sum(profit)/len(profit)
        if low:
            profit = 1+1-profit # This is for selling short
        return profit
    
    def run_trading_sim(self, n, date, period_len, show=True, low=False, 
        entropy=True, steps=3, e_high=0, company_prune=1, ignore_thr=1,
          use_meta_model=False, score_prop=0.25, track_wandb=0):
        if track_wandb:
            import wandb
            wandb.init(
                project=f'Stock Sim {datetime.date.today()}',
                config={
                    'mm_score_prop':score_prop,
                    'num_companies':n,
                    'cprop':company_prune
                },
                name=f'n={n}, mm={score_prop}, cprop={company_prune}'
            )
        self.c_prune = company_prune
        #ind = self.dataset.iterable_dates.index(date)
        ind = self.iterable_dates.index(date)
        #dates = self.dataset.iterable_dates[ind:ind+period_len]
        dates = self.iterable_dates[ind:ind+period_len]
        self.rank_profits_acc = [None for i in range(n)]
        profits = []
        counter = 0
        print(dates, 'num dates: ', len(dates))
        average_preds = []
        #bool_profit = []
        self.rank_profits_acc = [0 for i in range(n)]
        init_cum_money = self.cum_money
        for i in range(len(dates)):
            #if (i+self.n) % self.n == 0:
            profit = 1.0
            companies, pred = self.get_top_q_buy(n, dates[i], show=show, low=low, entropy=entropy,
                                                  steps=steps, e_high=e_high, use_meta_model=use_meta_model,
                                                  score_prop=score_prop)
            average_preds.append(pred)
            average_pred = sum(average_preds)/len(average_preds)
            thr_ind = int(len(average_preds)/4)
            threshold = sorted(average_preds)[thr_ind] if not low else sorted(average_preds)[-thr_ind]
            if ignore_thr:
                threshold = 1 if low else -1
            if ((pred > threshold and low==False) or (pred < threshold and low==True)) and i > 3:
                profit = self.get_date_profit(dates[i+1], companies, low=low,counter=counter)
                if profit is None:
                    profit = 1.0
                profits.append(profit)
                self.cum_money = self.cum_money*profit
                counter += 1
                if profit > 1:
                    self.bool_profit.append(1)
                else:
                    self.bool_profit.append(0)
            elif i <= 3:
                profit = self.get_date_profit(dates[i+1], companies, low=low,counter=counter)
                if profit is None:
                    profit = 1.0
                profits.append(profit)
                self.cum_money = self.cum_money*profit
                if profit > 1:
                    self.bool_profit.append(1)
                else:
                    self.bool_profit.append(0)
                counter += 1
            else:
                print(f'Skipping {dates[i]} because prediction {pred:.5f}% is less than desired threshold {average_pred:.5f}%')
                

                profits.append(profit)
                counter += 1
            if track_wandb:
                wandb.log({'Percent Change':self.cum_money/80000.0})
                wandb.log({'Probability of Profit':float(sum(self.bool_profit)/len(self.bool_profit))*100})
                wandb.log({'Point Profit':profit})
            print(f'\r{counter}: Point Profit: {profit:.4f} | Average Profit: {sum(profits)/len(profits):.4f} | Account: ${self.cum_money:.2f} | Probability of Profit: {float(sum(self.bool_profit)/len(self.bool_profit))*100:.4f}% | Percent Change: {self.cum_money/init_cum_money}')
        return profits
    
    def run_random_trading_sim(self, date, period_len, low=False):
        #self.c_prune = company_prune
        #ind = self.dataset.iterable_dates.index(date)
        self.cum_money = 80000
        #print(self.iterable_dates[0:3])
        #print(date)
        #date = datetime.date(2024,11,25)
        ind = self.iterable_dates.index(date)
        #dates = self.dataset.iterable_dates[ind:ind+period_len]
        dates = self.iterable_dates[ind:ind+period_len]
        profits = []
        counter = 0
        print(dates, 'num dates: ', len(dates))
        average_preds = []
        for i in range(len(dates)-1):
            #if (i+self.n) % self.n == 0:
            profit = 1.0
            companies = random.sample(self.companies,1)
            
            
            profit = self.get_date_profit(dates[i+1], companies, low=low)
            if profit is None:
                profit = 1.0
            profits.append(profit)
            self.cum_money = self.cum_money*profit
            counter += 1

            print(f'\r{counter}: Point Profit: {profit:.4f} | Average Profit: {sum(profits)/len(profits):.4f} | Account: ${self.cum_money:.2f}')
        return profits
    
    def run_random_trading_sims(self, date, period_len, num_sims):
        results = []
        #print(type(date))
        for i in range(num_sims):
            self.run_random_trading_sim(date,period_len)
            results.append(self.cum_money)
        results = sorted(results)
        results = torch.tensor(results)
        print(results)
        mean = results.mean()
        std = results.std()
        print(mean, std)
        
    def get_price_volume_bounds(self, date):
        '''
        This calculates the upper and lower bounds, as well as the mean and 
        std deviation of the total money traded for companies on the stock market
        on a given date (price * volume). Downstream, it is used to generate a z-score
        and unit cube placement of individual companies to get a relative indicator of
        market cash flow on the stock. 
        '''
        price_volumes = []
        for company in self.companies:
            try:
                day_data = self.dataset.data[date][company]
                price_volumes.append(day_data[0][-2]*day_data[0][-1])
            except Exception as e:
                pass
        #print(price_volumes[0].shape)
        t_price_volumes = torch.stack(price_volumes, 0)
        min = torch.min(t_price_volumes)
        max = torch.max(t_price_volumes)
        mean = torch.mean(t_price_volumes)
        std = torch.std(t_price_volumes)
        #print(mean, std)
        return min, max, mean, std
    
    def get_seq_pv_stats(self):
        #ind = self.dataset.iterable_dates.index(date)
        dates = self.dataset.iterable_dates
        mins, maxs, means, stds = [], [], [], []
        for date in dates:
            min, max, mean, std = self.get_price_volume_bounds(date)
            mins.append(min.item())
            maxs.append(max.item())
            means.append(mean.item())
            stds.append(std.item())
        min = sum(mins)/len(mins)
        max = sum(maxs)/len(maxs)
        mean = sum(means)/len(means)
        std = sum(stds)/len(stds)
        print(min, max, mean, std)
    

def main():

    l_model_pth = 'L1_Dist600_E40'
    l2_model_pth = 'mL2_Dist600_E16.5_E57'
    b_model_pth = 'Dist600__E12'

    model_pth = 'DistPred_50m_w2_E4'
    model_pth = 'DistPred_m_E7_'
    #model_pths = [model_pth, 'DistPred_m_E1', 'DistPred_m_E3']
    #model_pths = [model_pth, 'DistPred_m_E12_', 'DistPred_m_E3_L10']
    model_pths = ['DistPred_m_E22']
    #model = torch.load('full_model_E4')
    summaries_pth = 'i_summary_embs.pickle'
    sectors_pth = 'i_sector_dict'
    prices_pth = 'unnorm_price_series_i_-10y-2024-07-07__.pickle'
    #summaries_pth = 'a-z_summary_embs.pickle'
    #sectors_pth = 'a-z_sector_dict'
    #prices_pth = 'unnorm_price_series_a-z_-10y-2024-06-26__.pickle'
    data_dict_pth = 'DataDict_inf_f_10y'
    data_dict_pth = 'DataDict_inf_10y'
    use_meta_model = 0
    load_dataset = 1
    low = 0
    entropy = 0
    debug = 0
    e_high = 0
    top_n = 10
    n=1
    c_prune = 0.75
    curr_day = 1
    generate = 0
    ignore_thr = 1
    mm_prop = 1.0
    track_wandb = 1

    if curr_day:
        do_current_day_prediction(model_pths, n, top_n, low, entropy,debug=debug, 
            generate=generate, steps=1, c_prune=c_prune, ignore_thr=ignore_thr, 
            use_meta_model=use_meta_model, mm_prop=mm_prop, track_wandb=track_wandb)
    else:
        a = stock_inference(model_pth, summaries_pth, prices_pth, start_date=(2024, 1, 3), end_date=(2024, 6, 26), 
                n=n, seq_len=600, load_dataset=load_dataset, num_bins=320, sectors_pth=sectors_pth,
                debug=debug, data_dict_pth=data_dict_pth)
        #a.run_random_trading_sims()
        a.run_trading_sim(top_n, datetime.date(2024, 4, 5), 55, low=low, entropy=entropy, steps=2, e_high=e_high)

if __name__ == '__main__':
    main()