import torch 
from models import *
import datetime
import random 
import pickle 
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from contrastive_pretraining import gauss_normalize, prepare_ae_data
from tqdm import tqdm
import os
import wandb
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import *
from lion_pytorch import Lion
import numpy as np 
from transformers import Adafactor
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class GenerateDataDict:
    def __init__(self, bulk_prices_pth, summaries_pth, seq_len, 
                 save=True, j=True, save_name=None):
        self.bulk_prices = pic_load(bulk_prices_pth)
        self.summaries = pic_load(summaries_pth)
        self.seq_len = seq_len
        self.companies = list(self.bulk_prices.keys())
        print(len(self.companies))
        dates = []
        max_ind = 0
        max_len = 0
        for i in range(20):
            if i == 0:
                max_len = len(list(self.bulk_prices[self.companies[i]].keys()))
            else:
                if len(list(self.bulk_prices[self.companies[i]].keys())) >= max_len:
                    max_ind = i
                    max_len = len(list(self.bulk_prices[self.companies[i]].keys()))
            #print(max_len)
        self.dates = list(self.bulk_prices[self.companies[max_ind]].keys())
        self.iterable_dates = self.dates[seq_len:]
        if j:
            self.r_data = self.prepare_price_only()
        else:
            self.r_data = self.encode_data()
        if save:
            print('Saving Data Dict')
            if save_name:
                torch.save(self.r_data, save_name+'.pt')
                #with open(save_name, 'wb') as f:
                #    pickle.dump(self.r_data, f)
            else:
                torch.save(self.r_data, f'DataDict_{datetime.date.today()}.pt')
                #with open(f'DataDict_{datetime.date.today()}', 'wb') as f:
                #    pickle.dump(self.r_data, f)

    
    def get_ae_prices_chunk(self, date, company):
        ind = self.dates.index(date)
        dates = self.dates[ind-self.seq_len:ind]
        data = []
        for date in dates:
            try:
                data.append(self.bulk_prices[company][date])
            except Exception as e:
                return None
        a = torch.stack(data, dim=0).unsqueeze(0)
        return a

    def get_nor_seq_price_data(self, date, company):
        ind = self.dates.index(date)
        dates = self.dates[ind-self.seq_len:ind]    
        data = []
        for date in dates:
            data.append(self.bulk_prices[company][date])
        data = torch.stack(data, dim=1)
        return data 
    
    def encode_data(self):
        self.data = {}
        counter = 0
        top = len(self.iterable_dates)*len(self.companies)
        counter2 = 0
        with torch.no_grad():
            for date in self.iterable_dates:
                self.data[date] = {}
                for company in self.companies:
                    #try:
                    p = self.get_ae_prices_chunk(date, company)
                    try:
                        s = self.summaries[company]
                        a = self.bulk_prices[company][date].unsqueeze(0)
                    except Exception as e:
                        p = None
                    if p is None:
                        counter2+=1
                        pass
                    else:
                        s, p = prepare_ae_data([s, p])
                        if torch.isnan(p).any() or torch.isnan(s).any():
                            print(1)
                        if torch.isinf(p).any() or torch.isinf(s).any():
                            print(1)
                        d1 = self.encoder.encode(p, s) # Tensor of size 913
                        if counter %5000 == 0:
                            print(d1[0][0:10])
                        #d2 = self.get_nor_seq_price_data(date, company)
                        d1 = torch.cat((d1.cpu(), a), dim=1)
                        self.data[date][company] = d1
                        counter += 1
                        print(f'\r{counter}, {counter2} / {top}', end='')
        return self.data

    def prepare_price_only(self):
        self.data = {}
        counter = 0
        top = len(self.iterable_dates)*len(self.companies)
        counter2 = 0
        for date in self.iterable_dates:
            self.data[date] = {}
            for company in self.companies:
                #try:
                #p = self.get_ae_prices_chunk(date, company)
                try:
                    a = self.bulk_prices[company][date].unsqueeze(0)
                except Exception as e:
                    a = None
                if a is None:
                    counter2+=1
                    pass
                else:
                    '''
                    # S is dim 71, repeat it n times to match seq_len
                    n = int(self.seq_len/71)
                    s = s.repeat(n, 1).unsqueeze(1)
                    #print(s.shape)
                    d1 = torch.cat((s, p), dim=1)
                    #print(d1.shape)
                    if counter %5000 == 0:
                        print(d1[0][0:10])
                    #d2 = self.get_nor_seq_price_data(date, company)
                    '''
                    #s, p = prepare_ae_data([s, p])
                    day_of_the_week = date.isoweekday()
                    day_of_the_week = torch.nn.functional.one_hot(torch.tensor(day_of_the_week)-1, 5).unsqueeze(0)
                    #print(day_of_the_week.shape, a.shape)
                    b = torch.cat((a,day_of_the_week),dim=1)
                    self.data[date][company] = {'Price':a,"Price_With_Day":b}
                    counter += 1
                    print(f'\r{counter}, {counter2} / {top}', end='')
        return self.data

class QTrainingData(Dataset): 
    def __init__(self, summaries_pth, bulk_prices_pth, seq_len, n=5, k=1, 
        c_prop=0.5, full=True, layer_args=None, load=True, inference=None, 
        sectors_pth=str, data_slice_num=int, data_dict_pth=str,inf_company_keep_rate=1):
        self.data_slice_num = data_slice_num
        if layer_args is not None:
            mdl_pth = layer_args[0]
            self.seq_len = seq_len
        self.seq_len = seq_len
        self.i_keep = inf_company_keep_rate
        self.n = n # Trade period len
        self.k = k # How often to take a point from the seqence
        print('Loading Summaries...')
        self.summaries = pic_load(summaries_pth)
        print('Loading Sectors...')
        self.sectors_info_dict = pic_load(sectors_pth)
        if load:
            #self.data = pic_load('DataDict_inf_10y')
            print('Loading Data Dict...')
            self.data = torch.load(data_dict_pth+'.pt')
            #self.data = pic_load(data_dict_pth)
            #torch.save(self.data, data_dict_pth+'.pt')
        else:
            print('Using ', bulk_prices_pth, 'as ', data_dict_pth)
            self.data = GenerateDataDict(bulk_prices_pth, summaries_pth, seq_len, 
                                         save=True, save_name=f'raw_{data_dict_pth}')
            self.data = self.data.r_data
            torch.save(self.data, data_dict_pth+'.pt')
            #save_pickle(self.data, data_dict_pth)

        self.dates = list(self.data.keys())
        self.iterable_dates = self.dates[self.seq_len:]
        print('iterable dates: ', len(self.iterable_dates))
        self.companies = list(self.data[self.dates[0]].keys())
        self.date_chunks = {}
        for i in range(len(self.dates)-self.seq_len):
            self.date_chunks[self.dates[i+self.seq_len]] = self.dates[i:i+self.seq_len]
        self.dataloader = []
        self.relative_price_dict = {}
        self.generate_full_market_stats()
        self.get_sector_industry_classes()
        if full and (layer_args == None):
            self.prepare_dataset(c_prop)
        if full and (layer_args is not None):
            self.model = torch.load(mdl_pth).eval().to('cuda')
            #self.prepare_dataset_layer(c_prop)
        if inference is not None:
            print("Generating inference dataset...")
            self.inference_data = {}
            start, end = inference[0], inference[1]
            self.inference_dataset(start, end)
            #save_pickle(self.inference_data, 'InferenceDataset')

    def generate_full_market_stats(self, dim=5):
        self.pca_stats = {}
        print('Generating Market Stats')
        for date in tqdm(self.dates):
            pca = self.get_day_market_stats(date, dim)

            self.pca_stats[date] = pca
        
    def get_price_volume_bounds(self, date):
        price_volumes = []
        for company in self.companies:
            try:
                day_data = self.data[date][company]['Price']
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
    
    def get_company_chunk_data(self, company, date):
        dates = self.date_chunks[date] 
        try:
            comp_data = self.data[dates[0]][company]['Price_With_Day'].unsqueeze(0)
            pca_stats = self.pca_stats[dates[0]].unsqueeze(0)
            rel_data = self.relative_price_dict[dates[0]][company]
            #print(comp_data)
            for _date in dates[1:]:
                rel_data = torch.cat((rel_data, self.relative_price_dict[_date][company]), dim=0)
                comp_data = torch.cat((comp_data, self.data[_date][company]['Price_With_Day'].unsqueeze(0)), dim=0)
                pca_stats = torch.cat((pca_stats, self.pca_stats[_date].unsqueeze(0)), dim=0)
            #print(rel_data.shape, comp_data.shape, pca_stats.shape)
            comp_data = comp_data.unsqueeze(0)
            c = self.summaries[company]
            price = self.data[dates[-1]][company]['Price'][0][-2] # For computing target value
        except Exception as e:
            #print(e)
            return None
        return (comp_data, c, pca_stats, price, rel_data)

    def get_n_company_chunks(self, company, date):
        data_chunks = []
        ind = self.dates.index(date)
        for i in range(self.n):
            a = self.get_company_chunk_data(company, date)
            #print('n_chk', type(a))
            if a is not None:
                #print('yes', a)
                data_chunks.append(a)
            else:
                return None
            try:
                date = self.dates[ind+i]
            except Exception as e:
                break
        return data_chunks
    
    def get_n_company_prices(self, company, date):
        try:
            data = self.get_company_chunk_data(company, date)
            prices = []
            ind = self.dates.index(date)
            for i in range(self.n):
                date = self.dates[ind+i]
                prices.append(self.data[date][company]['Price'][0][-2])
            #print(data)
        except Exception as e:
            #print(e)
            return None
        if data is None:
            #print('Fuck')
            return None
        else:
            return (data, prices)        

    def get_abs_price_dim(self, price_seq, s):
        # This represents the price volume score relative to the other companies on the market day
        # Shape (batch, seq, dim)
        # 2.465073550667833 16111693133.838446 99981918.3880809 557781883.0235642
        _min, _max, mean, std = s[0], s[1], s[2], s[3]
        close = price_seq[:, :, -2]
        volume = price_seq[:, :, -1]
        price_volume = close*volume
        gauss_price_volume = (price_volume-mean)/std
        cube_price_volume = 2*(price_volume-_min)/(_max-_min)-1
        added_data = torch.cat((gauss_price_volume.unsqueeze(2), cube_price_volume.unsqueeze(2)), dim=2).to(torch.float32)
        return added_data
    
    def prepare_seq_data(self, data, s):
        data = data.squeeze(2)
        _data = set_nan_inf(data)
        _data = gauss_normalize(_data, dim=1)
        __data = cube_normalize(_data, dim=1)
        a_data = self.get_abs_price_dim(data, s)
        _data = torch.cat((_data, __data, a_data), dim=2)
        return _data

    def get_day_market_stats(self, date, pca_dim=5):
        stats = []
        ind = self.dates.index(date)
        if ind == 0:
            return None
        prev_date = self.dates[ind-1]
        self.relative_price_dict[date] = {}
        #sectors_change_dict = {}
        for company in self.companies:
            try:
                day_data = self.data[date][company]['Price']
                prev_day_data = self.data[prev_date][company]['Price']
                change = (day_data-prev_day_data)/prev_day_data
            except Exception:
                change = torch.zeros(5).unsqueeze(0)
                #change = torch.zeros(5)
            stats.append(change)
            self.relative_price_dict[date][company] = change
        stats = torch.stack(stats, 0)
        stats = stats.squeeze(1)
        stats = set_nan_inf(stats)
        #u, s, v = torch.pca_lowrank(stats, q=pca_dim) # 3000 x dim, dim, dimx5
        mean = torch.mean(stats,  0) # 1 x 5
        std = torch.std(stats, 0) # 2 x 5
        #r_data = torch.cat((v.flatten(), mean.flatten(), std.flatten()), dim=0) # 35 dimensional
        r_data = torch.cat((mean.flatten(), std.flatten()), dim=0)
        return r_data

    def get_sector_industry_classes(self):
        self.sectors = []
        self.industries = []
        for company, data in self.sectors_info_dict.items():
            sector, industry = data[0], data[1]
            if not (sector in self.sectors):
                self.sectors.append(sector)
            if not (industry in self.industries):
                self.industries.append(industry)
        self.num_sectors = len(self.sectors)
        self.num_industries = len(self.industries)
        print('Num sectors: ', self.num_sectors, 'Num Industries ', self.num_industries)
        
        self.sectors_sinusoidal_encoding = PositionalEncoding(d_model=64+52, max_len=self.num_sectors).encoding
        self.industry_sinusoidal_encoding = PositionalEncoding(d_model=52+52, max_len=self.num_industries).encoding
    
    def add_sector_industry_embedding(self, summary, company):
        do_corrected = 1
        sector, industry = self.sectors_info_dict[company][0], self.sectors_info_dict[company][1]
        sector_ind, industry_ind = self.sectors.index(sector), self.industries.index(industry)
        if do_corrected:
            try: 
                self.sector_one_hot_encoding = torch.nn.functional.one_hot(torch.tensor(sector_ind), 11)
                self.industry_one_hot_encoding = torch.nn.functional.one_hot(torch.tensor(industry_ind), 170)
            except Exception as e:
                self.sector_one_hot_encoding = torch.nn.functional.one_hot(torch.tensor(10), 11)
                self.industry_one_hot_encoding = torch.nn.functional.one_hot(torch.tensor(169), 170)
        else:
            self.sector_one_hot_encoding = torch.nn.functional.one_hot(torch.tensor(sector_ind), len(self.sectors))
            self.industry_one_hot_encoding = torch.nn.functional.one_hot(torch.tensor(industry_ind), len(self.industries))
        #sector_emb, industry_emb = self.sectors_sinusoidal_encoding[sector_ind, :], self.industry_sinusoidal_encoding[industry_ind, :]
        #final_emb = torch.cat((summary, sector_emb, industry_emb), dim=0).detach().cpu()
        app = torch.ones((988-768))
        summary = torch.cat((app, summary), dim=0)
        final_emb = summary.cpu()
        return final_emb

    def prepare_dataset(self, c_prop):
        counter = 0
        c_prop = int(c_prop*len(self.companies))
        print(len(self.companies), ' Companies Available')
        c_slice_ind = self.data_slice_num
        start = 0 + c_slice_ind*c_prop
        #start = 0
        companies = self.companies[start:start+c_prop]
        print(f'Generating Dataset ({len(companies)} companies)')
        for date in tqdm(self.iterable_dates):
            counter += 1
            s = self.get_price_volume_bounds(date)
            stats = self.pca_stats[date]
            companies = self.companies[start:start+c_prop]
            #companies = random.sample(self.companies, c_prop)
            #print('len companies', len(companies))
            #print(date)
            for i, company in enumerate(companies):
                if (counter-2) % self.k == 0: 
                    #print(f'\r{counter}/{int(len(self.iterable_dates)/self.k)}', end='')
                    all_data = self.get_n_company_prices(company, date)
                    r_data = []
                    if all_data is None:

                        pass
                    else:
                        data, prices = all_data[0], all_data[1]
                        prices = [torch.tensor(price).cpu() for price in prices]
                        _data = data[0]
                        summary = data[1].squeeze(0).cpu()
                        summary = self.add_sector_industry_embedding(summary, company)
                        sector_one_hot = self.sector_one_hot_encoding.repeat(600,1)
                        industry_one_hot = self.industry_one_hot_encoding.repeat(600,1)
                        stats = data[2]
                        price = data[3]
                        rel_data = data[4]
                        #print(_data.shape)
                        _data = self.prepare_seq_data(_data, s).squeeze(0) # Shape 1 x 300 x 12
                        #print(_data.shape)
                        _data = torch.cat((_data.cpu(), stats.cpu(), rel_data.cpu()), dim=1).to(device='cpu', dtype=torch.float32)
                        #print(_data.shape)
                        _data = torch.cat((_data.cpu(), sector_one_hot, industry_one_hot), dim=1).to(device='cpu', dtype=torch.float32)
                        #print(_data.shape)
                        #if i == 3:
                        #    print(_data.shape)
                        r_data.append([_data, summary, prices])
                        if len(data) == self.n:  
                            self.dataloader.append(r_data)

    def prepare_dataset_layer(self, c_prop):
        with torch.no_grad():
            counter = 0
            c_prop = int(c_prop*len(self.companies))
            for date in tqdm(self.iterable_dates):
                counter += 1
                s = self.get_price_volume_bounds(date)
                stats = self.pca_stats[date]
                companies = random.sample(self.companies, c_prop)
                for company in companies:
                    if (counter-2) % self.k == 0: 
                        print(f'\r{counter}/{int(len(self.iterable_dates)/self.k)}', end='')
                        data = self.get_n_company_chunks(company, date)
                        r_data = []
                        if data is None:
                            #print(counter)
                            pass
                        else:
                            for list_data in data:
                                #print('Tup shape:', _data[0].shape, _data[1].shape, _data[2].shape)
                                _data = list_data[0]
                                summary = list_data[1].cpu()
                                stats = list_data[2]
                                price = list_data[3]
                                rel_data = list_data[4]
                                _data = self.prepare_seq_data(_data, s).squeeze(0) # Shape 1 x 300 x 12
                                #print('1x350x12, 350x35?',_data.shape, stats.shape)
                                #print(_data.shape, stats.shape, rel_data.shape)
                                _data = torch.cat((_data.cpu(), stats.cpu(), rel_data.cpu()), dim=1).to(device='cpu', dtype=torch.float32).detach()
                                #print(_data.shape)
                                #print('1x350x47?:', _data.shape)
                                #print(_data.shape, summary.shape, price.shape)
                                price = torch.tensor(price).cpu().detach()
                                encoder_data = _data[0:200, :].to('cuda').unsqueeze(0)
                                extended_data = _data[200:, :]
                                #print(encoder_data.shape, extended_data.shape)
                                tran_act, out_act = self.model.encode(encoder_data, summary.to('cuda'))
                                tran_act = tran_act.cpu().squeeze(0)
                                extended_data = torch.cat((extended_data, tran_act), dim=0).detach()

                                r_data.append([extended_data, out_act.cpu(), summary, price])
                                #print(_data.shape, summary.shape, price.shape)
                            if len(data) == self.n:  
                                self.dataloader.append(r_data)

    def inference_dataset(self, start_date, end_date, layer=False):
        counter = 0
        company_keep_rate = self.i_keep

        self.companies = random.sample(self.companies,int(len(self.companies)*company_keep_rate))
        # For Debugging
        print('First dates: ', self.iterable_dates[0:3], 'Last dates: ', 
              self.iterable_dates[-3:], 'Num dates: ', len(self.iterable_dates),
              'num companies: ', len(self.companies))
    
        s_ind = self.iterable_dates.index(start_date)
        e_ind = self.iterable_dates.index(end_date)
        dates = self.iterable_dates[s_ind:e_ind]
        #companies = self.companies[0:int(len(self.companies)/4)]
        for date in tqdm(dates):
            counter += 1
            s = self.get_price_volume_bounds(date)
            stats = self.pca_stats[date]
            self.inference_data[date] = {}
            for company in self.companies:
                all_data = self.get_n_company_prices(company, date)
                if all_data is None:
                    pass
                else:
                    data, prices = all_data[0], all_data[1]
                    prices = [torch.tensor(price).cpu() for price in prices]
                    _data = data[0]
                    summary = data[1].squeeze(0).cpu()
                    summary = self.add_sector_industry_embedding(summary, company)
                    sector_one_hot = self.sector_one_hot_encoding.repeat(600,1)
                    industry_one_hot = self.industry_one_hot_encoding.repeat(600,1)
                    stats = data[2]
                    price = data[3]
                    rel_data = data[4]
                    #print(stats.shape)
                    #print(_data.shape)
                    _data = self.prepare_seq_data(_data, s).squeeze(0) # Shape 1 x 300 x 12
                    #print(_data.shape)
                    _data = torch.cat((_data.cpu(), stats.cpu(), rel_data.cpu()), dim=1).to(device='cpu', dtype=torch.float32)
                    #print(_data.shape)
                    _data = torch.cat((_data.cpu(), sector_one_hot, industry_one_hot), dim=1).to(device='cpu', dtype=torch.float32)
                    #print(_data.shape)
                    #print(_data.shape)
                    self.inference_data[date][company] = [_data, summary, price]


    def __len__(self):
        return len(self.dataloader)
    
    def __getitem__(self, ind):
        return self.dataloader[ind]

def convert_reward_to_one_hot(value, num_bins, bounds=0.2, correct=True):
    if correct:
        value = value-1
    scaled_value = (value + bounds) / (bounds*2) * (num_bins - 1)
    bin_index = int(scaled_value)   
    if bin_index > num_bins-1:
        bin_index = num_bins-1
    if bin_index < 0:
        bin_index = 0
    one_hot = torch.zeros(num_bins)
    one_hot[bin_index] = 1.0
    return one_hot

def _convert_reward_to_one_hot_batch(values, bin_edges, correct=False, with_smoothing=0, smoothing=0.2):
    """
    Converts continuous reward values into a one-hot encoding based on dynamic bin edges.

    Args:
        values (Tensor): Continuous reward values (batch_size,).
        bin_edges (Tensor): Edges of the bins (num_bins + 1,).
        correct (bool): If True, adjusts the input values.
        with_smoothing (int): If True, applies smoothing to the one-hot encoding.
        smoothing (float): Smoothing factor for label smoothing (default=0.2).

    Returns:
        Tensor: One-hot encoded rewards (batch_size, num_bins).
    """
    #bin_edges = torch.load('Bin_Edges_200')
    num_bins = len(bin_edges) - 1  # Number of bins derived from bin_edges
    
    # Adjust values if the correct flag is set
    if correct:
        values = values - 1
    
    # Compute bin indices for each value
    bin_indices = torch.bucketize(values, bin_edges, right=False) - 1  # bucketize returns 1-based indices
    bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)  # Ensure indices are within [0, num_bins-1]
    
    # Create one-hot encoding
    one_hot = torch.zeros(values.size(0), num_bins, device=values.device)
    one_hot.scatter_(1, bin_indices.unsqueeze(1), 1.0)
    
    if with_smoothing:
        smooth_mass = smoothing / 2.0  # Spread smoothing equally to neighbors
        center_mass = 1.0 - smoothing

        # Reset the center bin with smoothed value
        one_hot = one_hot * center_mass

        # Add smoothing to neighboring bins
        for i in range(values.size(0)):
            center_bin = bin_indices[i]
            if center_bin > 0:  # Left neighbor
                one_hot[i, center_bin - 1] += smooth_mass
            if center_bin < num_bins - 1:  # Right neighbor
                one_hot[i, center_bin + 1] += smooth_mass

    return one_hot

def convert_reward_to_one_hot_batch(values, num_bins, bounds=0.2, correct=True, with_smoothing=0, smoothing=0.2):
    # Adjust values if correct flag is set
    if correct:
        values = values - 1
    
    # Scale values to bin indices
    scaled_values = (values + bounds) / (bounds * 2) * (num_bins - 1)
    bin_indices = scaled_values.long()
    
    # Clip bin indices to be within [0, num_bins-1]
    bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)
    

    # Create one-hot encoding
    #print(values.shape)
    one_hot = torch.zeros(values.size(0), num_bins, device=values.device)
    one_hot.scatter_(1, bin_indices.unsqueeze(1), 1.0)
    if with_smoothing:
        smooth_mass = smoothing / 2.0  # Spread smoothing equally to neighbors
        center_mass = 1.0 - smoothing

        # Reset the center bin with smoothed value
        one_hot = one_hot * center_mass

        # Add smoothing to neighboring bins
        for i in range(values.size(0)):
            center_bin = bin_indices[i]
            if center_bin > 0:  # Left neighbor
                one_hot[i, center_bin - 1] += smooth_mass
            if center_bin < num_bins - 1:  # Right neighbor
                one_hot[i, center_bin + 1] += smooth_mass



    return one_hot

def smooth_one_hot_reward(targets, num_bins, smoothing=0.1):
    """
    Converts regression targets into a smoothed one-hot encoding across bins.

    Args:
        targets (Tensor): Tensor of shape (batch_size,) with continuous values.
        num_bins (int): Total number of bins to discretize the targets.
        smoothing (float): Smoothing factor for label smoothing (0 means pure one-hot).

    Returns:
        Tensor: Smoothed one-hot target of shape (batch_size, num_bins).
    """
    assert 0.0 <= smoothing <= 1.0, "Smoothing factor must be between 0 and 1"

    # Compute the bin index for each target
    bin_width = 1.0 / num_bins
    target_bins = torch.clamp((targets / bin_width).long(), 0, num_bins - 1)

    # Initialize the smoothed target distribution
    batch_size = targets.size(0)
    smoothed_targets = torch.zeros(batch_size, num_bins, device=targets.device)

    # Define smoothing weights
    smooth_mass = smoothing / 2.0  # Spread smoothing equally to neighbors
    center_mass = 1.0 - smoothing  # Remaining weight on the target bin

    for i in range(batch_size):
        # Get the target bin index
        target_bin = target_bins[i]

        # Distribute mass to center, left neighbor, and right neighbor
        smoothed_targets[i, target_bin] += center_mass
        if target_bin > 0:  # Left neighbor
            smoothed_targets[i, target_bin - 1] += smooth_mass
        if target_bin < num_bins - 1:  # Right neighbor
            smoothed_targets[i, target_bin + 1] += smooth_mass

    return smoothed_targets

def prepare_direct_pred_data(data, device='cuda', dtype=torch.float32):
    data = data.squeeze(2)
    _data = set_nan_inf(data).to(device)
    _data = gauss_normalize(_data, dim=1)
    __data = cube_normalize(_data, dim=1)
    a_data = get_abs_price_dim(data).to(device)
    _data = torch.cat((_data, __data, a_data), dim=2).to(device, dtype=dtype)
    return _data

def generate_EL2N_pruned_dataset(dataset_pth, proportion, model_pth, device='cuda',
                                  dtype=torch.float32, num_bins=320):
    model = torch.load(model_pth).eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    dataset = pic_load(dataset_pth)
    losses = [None for i in range(len(dataset))]

    # Calculate the losses for the dataset
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataset)):
            target = None
            data = data[0]
            data, summary, prices = data[0], data[1], data[2]
            data = set_nan_inf(data)
            
            # For light data augmentation
            #data = gaussian_noise(data, 1e-6).unsqueeze(0) # Unsqueeze to add batch dim
            #summary = gaussian_noise(summary, 1e-7).unsqueeze(0)

            pred = model(data.to(device, dtype=dtype), summary.to(device, dtype=dtype))
            buy_price = prices[0]
            for sell_price in prices[1:]:
                profit = [sell_price/buy_price]
                if target == None:
                    target = [convert_reward_to_one_hot_batch(item, num_bins) for item in profit]
                else:
                    _target = [convert_reward_to_one_hot_batch(item, num_bins) for item in profit]
                    _target = torch.stack(_target)
                    target = torch.cat((target, _target), dim=0)

            target = torch.permute(target, (1, 2, 0)) # b x 100 x 4
            ti = target.size(2)
            for i in range(ti):
                if i == 0:
                    loss = loss_fn(pred[:,:,i], target[:,:,i].to(device))
                else:
                    loss += loss_fn(pred[:,:,i], target[:,:,i].to(device))
            
            losses[i] = loss.item()
    
    # Initialize new list with sorted losses
    sorted_losses = sorted(losses)

    # Get number of elements to keep
    num_items_to_keep = int(len(dataset)*proportion)
    
    # Get the highest loss values corresponding to the pruned dataset
    losses_to_keep = sorted_losses[-num_items_to_keep:]

    # Get the indices of the pruned dataset
    indices_to_keep = [losses.index(item) for item in losses_to_keep]

    # Generate the list
    pruned_data = [dataset[item] for item in indices_to_keep]
    
    # Save the data
    save_pickle(pruned_data, f'{proportion}P_EL2N_{dataset_pth}')

def generate_outlier_dataset(dataset_pth, proportion):
    '''
    This function generates a dataset which contains the data points corresponding to the largest absolute
    changes in price.  This is doesn't take the model's prediction into account, only the data itself.
    '''
    print(f'Generating {proportion} outlier on {dataset_pth}')
    dataset = pic_load(dataset_pth)
    price_change = [None for i in range(len(dataset))]
    print('Num data points: ', len(price_change))
    for i, data in tqdm(enumerate(dataset)):
        profit = []
        data = data[0]
        _data = data[0]
        prices = data[1]
        buy_price = prices[0]
        for sell_price in prices[1:]:
            profit.append(sell_price.item()/buy_price.item())
        change = sum(item for item in profit)
        price_change[i] = change
    
    num_items_to_keep = int(len(dataset)*proportion)
    sorted_price_change = sorted(price_change)
    keep_price_changes = sorted_price_change[-num_items_to_keep:]
    ind_keep_changes = [price_change.index(item) for item in tqdm(keep_price_changes)]
    pruned_data = [dataset[item] for item in tqdm(ind_keep_changes)]
    print('Saving ', len(pruned_data), 'data points')
    print('Sample point: ', pruned_data[0])
    save_pickle(pruned_data, f'{proportion}P_Outlier_{dataset_pth}')

def generate_model_outlier_dataset(dataset_pth, proportion, model_pth, device='cuda', 
                                   dtype=torch.float32, num_bins=320, target_n=2):
    '''
    This function generates a dataset of points which the model predicts will have the largest absolute change. These are the
    points which it will eventually choose to buy, so it they are the one's most crucial for the model to predict accurately.
    '''

    model = torch.load(model_pth).eval()
    dataset = pic_load(dataset_pth)
    mean_predictions = [None for i in range(len(dataset))]

    # Calculate the outliers for the dataset
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataset)):
            _data = data[0]
            prices = data[1]
            pred, data = _data[0].unsqueeze(0), _data[1].unsqueeze(0)
            #pred, data = tup[0], tup[1]
            #print(data.shape, pred.shape)
            data = set_nan_inf(data)

            pred = model(data.to(device, dtype=dtype), pred.to(device, dtype=dtype)).squeeze(0)
            mean_pred = get_expected_price(pred[:,0], num_bins=num_bins, full_dist=False)
            mean_predictions[i] = abs(mean_pred)
    
    # Initialize new list with sorted predictions
    sorted_predictions = sorted(mean_predictions)

    # Get number of elements to keep
    num_items_to_keep = int(len(dataset)*proportion)
    
    # Get the highest absolute prediction values corresponding to the pruned dataset
    items_to_keep = sorted_predictions[-num_items_to_keep:]

    # Get the indices of the pruned dataset
    indices_to_keep = [mean_predictions.index(item) for item in tqdm(items_to_keep)]

    # Generate the list
    pruned_data = [dataset[item] for item in tqdm(indices_to_keep)]
    
    # Save the data
    save_pickle(pruned_data, f'{proportion}P_Out_{dataset_pth}')

def generate_naive_layer_dataset(dataset_pth, model_pth, device='cuda', 
                                 dtype=torch.float32, data_stack=True, num_chunks=1, half=0):
    '''
    Generates the prediction and transformer output of the model over a dataset, and stores it in a 
    new dataset for downstream greedy training.

    Args:
        dataset_pth (str): The path to the dataset, must be stored as a pickle file
        model_pth (str): A path to the PyTorch Model 
        data_stack (bool): If doing stack, the original data is stacked onthe transformer activation
                    in the embedding dimension, to provide as much information as possible to the next module.
        num_chunks (int): The number of seporate datasets to divide the new dataset into.
        half (0 or 1): Which half of the dataset to encode (This is necessary so the CPU doesn't run out 
                       of memory)
    '''

    model = torch.load(model_pth).eval()
    dataset = pic_load(dataset_pth)
    if half == 0:
        dataset = dataset[:int(len(dataset)/2)]
    else:
        dataset = dataset[int(len(dataset)/2):]
    new_dataset = []

    # Run the dataset through the model
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataset)):
            #target = None
            data = data[0]
            data, summary, prices = data[0].unsqueeze(0).to(device, dtype=dtype), data[1].to(device, dtype=dtype), data[2]
            data = set_nan_inf(data)

            pred = model(data, summary).cpu()
            t_activation = model.transformer(data, summary)
            
            if data_stack:
            # This is roughly equivelent to the first part of the model's forward method
                data = torch.flip(data,[1])
                data = data + model.pos_encode(data)
            
                # Reshape this to (batch, _, 52) so it can be appended to the end of the sequence
                summary = torch.reshape(summary, (1, 19, 52))
                
                # Add these data points to existing seqence
                data = torch.cat((data, summary), dim=1)

            t_activation = torch.cat((t_activation, data), dim=2).cpu()
            pred, t_activation = pred.squeeze(0), t_activation.squeeze(0)
            new_dataset.append(((pred, t_activation), prices))
    i = len(new_dataset)
    sub_dataset_size = int(i/num_chunks)
    for i in range(num_chunks):
        ind = i*sub_dataset_size
        sub_dataset = new_dataset[ind:ind+sub_dataset_size]
        save_pickle(sub_dataset, f'L1_H{half}_C{i}_{dataset_pth}')

def generate_naive_layer_2_dataset(dataset_pth, model_pth, device='cuda', 
                                 dtype=torch.float32, num_chunks=1, half=0):
    '''
    Generates the prediction and transformer output of the model over a dataset, and stores it in a 
    new dataset for downstream greedy training.

    Args:
        dataset_pth (str): The path to the dataset, must be stored as a pickle file
        model_pth (str): A path to the PyTorch Model 
        data_stack (bool): If doing stack, the original data is stacked onthe transformer activation
                    in the embedding dimension, to provide as much information as possible to the next module.
        num_chunks (int): The number of seporate datasets to divide the new dataset into.
        half (0 or 1): Which half of the dataset to encode (This is necessary so the CPU doesn't run out 
                       of memory)
    '''

    model = torch.load(model_pth).eval()
    dataset = pic_load(dataset_pth)
    if half == 0:
        dataset = dataset[:int(len(dataset)/2)]
    else:
        dataset = dataset[int(len(dataset)/2):]
    new_dataset = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataset)):
            _data = data[0]
            prices = data[1]
            pred, data = _data[0].unsqueeze(0), _data[1].unsqueeze(0)
            #print(data[:,0,0:51], data[:,0,52:])
            # t_act is 0:52, base data is 52:104
            base_data = data[:,:,:52]
            #print(base_data.shape, data.shape)
            data = set_nan_inf(data)
            pred = model(data.to(device, dtype=dtype), pred.to(device, dtype=dtype)).squeeze(0).cpu()
            t_activation = model.transformer(data.to(device, dtype=dtype)).cpu()
            t_activation = torch.cat((base_data, t_activation), dim=2).cpu()
            #print(pred.shape, t_activation.shape)
            new_dataset.append(((pred, t_activation), prices))
    i = len(new_dataset)
    sub_dataset_size = int(i/num_chunks)
    for i in range(num_chunks):
        ind = i*sub_dataset_size
        sub_dataset = new_dataset[ind:ind+sub_dataset_size]
        save_pickle(sub_dataset, f'L2_H{half}_C{i}_{dataset_pth}')

def convert_dataset_np(dataset_pth):
    data = pic_load(dataset_pth)
    data = np.array(data, dtype=object)
    save_pickle(data, f'np_{dataset_pth}')

def warmup_lambda(current_step):
    warmup_steps = 1000
    if current_step < warmup_steps:
        return current_step / warmup_steps
    return 1.0

def Train_Dist_Direct_Predictor(model, epochs, save_name, lr, num_bins, t_max, 
                        weight_decay, grd_nrm, misc, device='cuda', dtype=torch.float32, 
                        bounds=0.2,thr=1000,load_optim=True,grad_acc_steps=1, use_warmup=1,
                        train_prop=0.2):
    '''
    This is the primary training function, operating on the base layer. It takes in a model, list of dataset paths,
    various training parameters, and runs training on a direct distributional stock price prediction. 
    Args:
        model (nn.Module): the Pytorch Model to be trained
        epochs (int): Maximum epochs before terminating training
        save_name (str): Name for saving the model weights
        lr (float): learning rate
        num_bins (int): The number of bins the model uses to model its distribution
        t_max (int): The number of steps for one iteration of the Cosine Annealing learning rate scheduler
        weight_decay (float): Optimizer weight decay strength (L2 regularization)
        grd_nrm (float): Max gradient norm for clipping
        misc (iter): an iterable containing
                misc[0] -> batch size (int)
                misc[1] -> ndays (int): 
                                The number of market days in the future for the model to predict
                misc[2] -> data_pths (iter (containing strings)): 
                                An iterable containing the paths of the datasets to be trained on (stored as pickle files)

                misc[3] -> save_rate (int):
                                How often the model weights are saved (every _ epochs)
        device: The device to train on (i.e 'cuda:0')
        dtype: The dtype to cast the model weights and data
    '''
    batch_size, n_days, data_pths, save_rate = misc[0], misc[1], misc[2], misc[3]
    #optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    #optim = Adafactor(model.parameters(), lr, weight_decay=weight_decay, scale_parameter=False, relative_step=False)
    extra_params = list(model.pos_emb.parameters()) + list(model.linear_in.parameters())
    lr_configs = linear_layer_lr_scaling(model.layers, lr, lr*1.5, extra_params=extra_params)
    optim = Lion(lr_configs,betas=(0.95, 0.98),weight_decay=weight_decay*10,use_triton=True)
    model.train()
    
    loss_fn = torch.nn.CrossEntropyLoss()

    if load_optim:
        optim.load_state_dict(torch.load('optim.pth'))

    scaler = torch.cuda.amp.GradScaler()
    #scaler.load_state_dict(torch.load('scaler.pth'))
    bin_edges = torch.load(f'Bin_Edges_{num_bins}')
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}:")
        it = 0
        for pth in data_pths:
            if len(data_pths) == 1 and epoch != 0:
                pbar = tqdm(dataloader)
                running_loss = 0
                steps = 0
                
            else:
                d = pic_load(pth)
                dataloader = DataLoader(d, batch_size, shuffle=True, pin_memory=1)
                pbar = tqdm(dataloader)
                running_loss = 0
                steps = 0
                bp_steps = 0
                ema_loss = 20
                it += 1
                print(f'\nUsing dataset: {pth} | ({it}/{len(data_pths)})\n')
            if epoch == 0 and use_warmup:
                scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lambda)
            #else:
            #    del scheduler
            for i, data in enumerate(dataloader):
                    target = None
                    data = data[0]
                    data, summary, prices = data[0], data[1], data[2]
                    #print(data.shape)
                    data = set_nan_inf(data)
                    summary = set_nan_inf(summary)
                    # For light data augmentation
                    data[:,:,15:] = gaussian_noise(data[:,:,15:], 7e-2)
                    #data[:,:,-180:] = gaussian_noise(data[:,:,-180:], 5e-2)
                    #summary = gaussian_noise(summary, 5e-5)
                    
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        #try:
                        pred = model(data.to(device, dtype=dtype), summary.to(device, dtype=dtype))
                        buy_price = prices[0]
                        for sell_price in prices[1:]:
                            profit = [sell_price/buy_price]
                            if target == None:
                                target = [_convert_reward_to_one_hot_batch(item, bin_edges) for item in profit]
                                #target = [smooth_one_hot_reward(item, num_bins, smoothing=0.2) for item in profit]
                                target = torch.stack(target)
                            else:
                                _target = [_convert_reward_to_one_hot_batch(item, bin_edges) for item in profit]
                                _target = torch.stack(_target)
                                target = torch.cat((target, _target), dim=0)
                        target = torch.permute(target, (1, 2, 0)) # b x 100 x 4
                        ti = target.size(2)
                        for i in range(ti):
                            if i == 0:
                                #if steps == 0:
                                    #print(torch.sum(pred[:,:,i]))
                                loss = loss_fn(pred[:,:,i], target[:,:,i].to(device))
                                #loss = compute_loss_with_gaussian_regularization(pred[:,:,i], target[:,:,i].to(device),loss_fn)
                            else:
                                loss += loss_fn(pred[:,:,i], target[:,:,i].to(device))
                                #loss += compute_loss_with_gaussian_regularization(pred[:,:,i], target[:,:,i].to(device),loss_fn)
                        t_loss = loss.item()
                        running_loss += t_loss
                        steps += 1
                        avg_loss = running_loss/steps

                        if (not torch.isnan(loss).any() and loss.item() > 0):
                            
                            #if t_loss >= running_loss/(steps+1):
                            #print(1)
                            scaler.scale(loss).backward()
                            bp_steps += 1
                            
                            if loss.item() < thr and bp_steps % grad_acc_steps == 0:
                                scaler.unscale_(optim)
                                torch.nn.utils.clip_grad_norm_(model.parameters(), grd_nrm)
                                
                                # Step the optimizer only if gradients were computed
                                if any(param.grad is not None for param in model.parameters()):
                                    scaler.step(optim)
                                    scaler.update()
                                    optim.zero_grad()
                                    if epoch == 0 and use_warmup:
                                        scheduler.step()
                            #else:
                            #    optim.zero_grad()
                        
                        #except Exception as e:
                        #    print(e)
                    if steps % 1 == 0 and steps > 0:
                        a = f'Running Epoch Loss: {(running_loss/steps):.6f} Epoch Steps: {steps}/{len(dataloader)}  Step Loss: {t_loss:.6f}'
                        pbar.set_postfix_str(a)
                        pbar.update(1)
            torch.save(model, save_name+f"E{epoch+1}")
            torch.save(optim.state_dict(), 'optim.pth')
            torch.save(scaler, "scaler.pth")
            print(f'saved model ({steps})')

def Train_Dist_Direct_Predictor_SAM(model, epochs, save_name, lr, num_bins, t_max, 
                        weight_decay, grd_nrm, misc, device='cuda', dtype=torch.float32, 
                        bounds=0.2,thr=100000000,load_optim=True,grad_acc_steps=1, use_warmup=1,
                        train_prop=0.2):
    '''
    This is the primary training function, operating on the base layer. It takes in a model, list of dataset paths,
    various training parameters, and runs training on a direct distributional stock price prediction. 
    Args:
        model (nn.Module): the Pytorch Model to be trained
        epochs (int): Maximum epochs before terminating training
        save_name (str): Name for saving the model weights
        lr (float): learning rate
        num_bins (int): The number of bins the model uses to model its distribution
        t_max (int): The number of steps for one iteration of the Cosine Annealing learning rate scheduler
        weight_decay (float): Optimizer weight decay strength (L2 regularization)
        grd_nrm (float): Max gradient norm for clipping
        misc (iter): an iterable containing
                misc[0] -> batch size (int)
                misc[1] -> ndays (int): 
                                The number of market days in the future for the model to predict
                misc[2] -> data_pths (iter (containing strings)): 
                                An iterable containing the paths of the datasets to be trained on (stored as pickle files)

                misc[3] -> save_rate (int):
                                How often the model weights are saved (every _ epochs)
        device: The device to train on (i.e 'cuda:0')
        dtype: The dtype to cast the model weights and data
    '''
    batch_size, n_days, data_pths, save_rate = misc[0], misc[1], misc[2], misc[3]
    extra_params = list(model.pos_emb.parameters()) + list(model.linear_in.parameters())
    lr_configs = linear_layer_lr_scaling(model.layers, lr, lr*1.5, extra_params=extra_params)
    #optim = Lion(lr_configs,betas=(0.95, 0.98),weight_decay=weight_decay*10,use_triton=True)
    optim = Lion
    optim = SAM(model.parameters(), optim, rho=0.05, betas=(0.95,0.98), weight_decay=weight_decay*10,use_triton=True)
    model.train()
    
    loss_fn = torch.nn.CrossEntropyLoss()

    if load_optim:
        optim.load_state_dict(torch.load('optim.pth'))

    scaler = torch.cuda.amp.GradScaler()
    #scaler.load_state_dict(torch.load('scaler.pth'))
    bin_edges = torch.load(f'Bin_Edges_{num_bins}')
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}:")
        it = 0
        for pth in data_pths:
            if len(data_pths) == 1 and epoch != 0:
                pbar = tqdm(dataloader)
                running_loss = 0
                steps = 0
                
            else:
                d = pic_load(pth)
                dataloader = DataLoader(d, batch_size, shuffle=True, pin_memory=1)
                pbar = tqdm(dataloader)
                running_loss = 0
                steps = 0
                bp_steps = 0
                ema_loss = 20
                it += 1
                print(f'\nUsing dataset: {pth} | ({it}/{len(data_pths)})\n')
            if epoch == 0 and use_warmup:
                scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lambda)
            #else:
            #    del scheduler
            batch_data = []
            for i, data in enumerate(dataloader):
                    target = None
                    data = data[0]
                    data, summary, prices = data[0], data[1], data[2]
                    #print(data.shape)
                    data = set_nan_inf(data)
                    summary = set_nan_inf(summary)
                    # For light data augmentation
                    data[:,:,15:] = gaussian_noise(data[:,:,15:], 7e-2)
                    #data[:,:,-180:] = gaussian_noise(data[:,:,-180:], 5e-2)
                    #summary = gaussian_noise(summary, 5e-5)
                    
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        #try:
                        pred = model(data.to(device, dtype=dtype), summary.to(device, dtype=dtype))
                        buy_price = prices[0]
                        for sell_price in prices[1:]:
                            profit = [sell_price/buy_price]
                            if target == None:
                                target = [_convert_reward_to_one_hot_batch(item, bin_edges) for item in profit]
                                #target = [smooth_one_hot_reward(item, num_bins, smoothing=0.2) for item in profit]
                                target = torch.stack(target)
                            else:
                                _target = [_convert_reward_to_one_hot_batch(item, bin_edges) for item in profit]
                                _target = torch.stack(_target)
                                target = torch.cat((target, _target), dim=0)
                        target = torch.permute(target, (1, 2, 0)) # b x 100 x 4
                        ti = target.size(2)
                        for i in range(ti):
                            if i == 0:
                                #if steps == 0:
                                    #print(torch.sum(pred[:,:,i]))
                                loss = loss_fn(pred[:,:,i], target[:,:,i].to(device))
                                
                                #loss = compute_loss_with_gaussian_regularization(pred[:,:,i], target[:,:,i].to(device),loss_fn)
                            else:
                                loss += loss_fn(pred[:,:,i], target[:,:,i].to(device))

                                #loss += compute_loss_with_gaussian_regularization(pred[:,:,i], target[:,:,i].to(device),loss_fn)
                    batch_data.append((data.detach(),summary.detach(),target.detach()))
                    t_loss = loss.item()
                    running_loss += t_loss
                    steps += 1
                    #print(torch.isnan(pred).any(), torch.isinf(pred).any(),pred.max(),pred.min())

                    if (not torch.isnan(loss).any() and loss.item() > 0):
                        
                        scaler.scale(loss).backward()
                        bp_steps += 1
                        
                        if loss.item() < thr and bp_steps % grad_acc_steps == 0:
                            scaler.unscale_(optim.base_optimizer)
                            #torch.nn.utils.clip_grad_norm_(model.parameters(), grd_nrm)
                            optim.first_step(zero_grad=True)
                            for acc_step in range(grad_acc_steps):  # Iterate through accumulated batches
                                data,summary,target = batch_data[acc_step]  # Re-fetch mini-batches
                                #print(1)
                                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                                    pred = model(data.to(device, dtype=dtype), summary.to(device, dtype=dtype))
                                    ti = target.size(2)
                                    for i in range(ti):
                                        if i == 0:
                                            loss = loss_fn(pred[:,:,i], target[:,:,i].to(device))
                                        else:
                                            loss += loss_fn(pred[:,:,i], target[:,:,i].to(device))
                                scaler.scale(loss).backward()
                            #scaler.unscale_(optim.base_optimizer)
                            #torch.nn.utils.clip_grad_norm_(model.parameters(), grd_nrm)
                            batch_data = []
                            # Step the optimizer only if gradients were computed
                            if any(param.grad is not None for param in model.parameters()):
                                #scaler.step(optim)
                                optim.second_step(zero_grad=True)
                                scaler.update()
                                optim.zero_grad()
                                if epoch == 0 and use_warmup:
                                    scheduler.step()
                        #else:
                        #    optim.zero_grad()
                    
                    #except Exception as e:
                    #    print(e)
                    if steps % 1 == 0 and steps > 0:
                        a = f'Running Epoch Loss: {(running_loss/steps):.6f} Epoch Steps: {steps}/{len(dataloader)}  Step Loss: {t_loss:.6f}'
                        pbar.set_postfix_str(a)
                        pbar.update(1)
            torch.save(model, save_name+f"E{epoch+1}")
            torch.save(optim.state_dict(), 'optim.pth')
            torch.save(scaler, "scaler.pth")
            print(f'saved model ({steps})')

def Train_Layer_Dist_Direct_Predictor(model, epochs, save_name, lr, num_bins, t_max, 
                        weight_decay, grd_nrm, misc, device='cuda', dtype=torch.float32):
    '''
    This is the primary training function, operating on the base layer. It takes in a model, list of dataset paths,
    various training parameters, and runs training on a direct distributional stock price prediction. 
    Args:
        model (nn.Module): the Pytorch Model to be trained
        epochs (int): Maximum epochs before terminating training
        save_name (str): Name for saving the model weights
        lr (float): learning rate
        num_bins (int): The number of bins the model uses to model its distribution
        t_max (int): The number of steps for one iteration of the Cosine Annealing learning rate scheduler
        weight_decay (float): Optimizer weight decay strength (L2 regularization)
        grd_nrm (float): Max gradient norm for clipping
        misc (iter): an iterable containing
                misc[0] -> batch size (int)
                misc[1] -> ndays (int): 
                                The number of market days in the future for the model to predict
                misc[2] -> data_pths (iter (containing strings)): 
                                An iterable containing the paths of the datasets to be trained on (stored as pickle files)
                misc[3] -> save_rate (int):
                                How often the model weights are saved (every _ epochs)
                misc[4] -> noise_level (float): 
                                The variance of the noise to add to the input data
                misc[5] -> full_stack (bool):
                                Whether or not to train the full network, or one layer (False is layerwise). In
                                practice, this corresponds to using the AdamW or Lion Optimizer.
        device: The device to train on (i.e 'cuda:0')
        dtype: The dtype to cast the model weights and data
    '''
    for param in model.parameters():
        param = param.to(dtype)
    batch_size, n_days, data_pths, save_rate, noise_level, full_stack = misc[0], misc[1], misc[2], misc[3], misc[4], misc[5]
    
    #optim = Adafactor(model.parameters(), lr, weight_decay=weight_decay, scale_parameter=False, relative_step=False)
    if full_stack:
        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optim = Lion(model.parameters(), lr=lr/3, weight_decay=weight_decay*3)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=t_max, eta_min=lr/10)
    loss_fn = torch.nn.CrossEntropyLoss()

    #optim.load_state_dict(torch.load('optim.pth'))

    scaler = torch.cuda.amp.GradScaler()
    #scaler.load_state_dict(torch.load('scaler.pth'))
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}:")
        j = 0
        for pth in data_pths:
            j += 1
            if len(data_pths) == 1 and epoch != 0:
                pbar = tqdm(dataloader)
                running_loss = 0
                steps = 0
            else:
                try:
                    d = pic_load(pth)
                    dataloader = DataLoader(d, batch_size, shuffle=True, pin_memory=0)
                    pbar = tqdm(dataloader)
                    running_loss = 0
                    steps = 0
                except Exception as e:
                    print(e)
                print(f'\nUsing dataset: {pth} ({j}/{len(data_pths)})\n')
            for i, data in enumerate(dataloader):
                    target = None
                    _data = data[0]
                    prices = data[1]
                    pred, data = _data[0], _data[1]
                    #pred, data = tup[0], tup[1]
                    data = set_nan_inf(data.squeeze(1))
                    
                    # For data augmentation
                    data = gaussian_noise(data, noise_level)

                    with torch.autocast(device_type=device, dtype=torch.float32):
                        pred = model(data.to(device, dtype=dtype), pred.to(device, dtype=dtype))
                        buy_price = prices[0]
                        for sell_price in prices[1:]:
                            profit = [sell_price/buy_price]
                            if target == None:
                                target = [convert_reward_to_one_hot_batch(item, num_bins) for item in profit]
                                target = torch.stack(target)
                            else:
                                _target = [convert_reward_to_one_hot_batch(item, num_bins) for item in profit]
                                _target = torch.stack(_target)
                                target = torch.cat((target, _target), dim=0)
                        target = torch.permute(target, (1, 2, 0)) # b x 100 x 4
                        ti = target.size(2)
                        for i in range(ti):
                            if i == 0:
                                loss = loss_fn(pred[:,:,i], target[:,:,i].to(device))
                            else:
                                loss += loss_fn(pred[:,:,i], target[:,:,i].to(device))
                    
                    if not torch.isnan(loss).any() and loss.item() > 0:
                        t_loss = loss.item()
                        #loss.backward()

                        # For evening out the predictions
                        loss = (loss/avg_loss)**2
                        scaler.scale(loss).backward()
                        scaler.unscale_(optim)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grd_nrm)
                        scaler.step(optim)
                        scaler.update()
                        #optim.step()
                        #scheduler.step()
                        optim.zero_grad()
                        running_loss += t_loss
                    steps += 1
                    avg_loss = running_loss/steps
                    if steps % 1 == 0 and steps > 0:
                        a = f'Running Epoch Loss: {(running_loss/steps):.6f} Epoch Steps: {steps}/{len(dataloader)}  Step Loss: {t_loss:.6f}'
                        pbar.set_postfix_str(a)
                        pbar.update(1)
            torch.save(model, save_name+f"E{epoch+1}")
            torch.save(optim.state_dict(), 'optim.pth')
            torch.save(scaler, "scaler.pth")
            print(f'saved model ({steps})')

def dataset_k_means(dataset, n_clusters=10, batch=0):
    print(type(dataset))
    #data = data[0]
    #data, summary, prices = data[0], data[1], data[2]
    #data = set_nan_inf(data)
    batch_len = int(len(dataset)/3)
    dataset = [torch.cat((set_nan_inf(item[0][0].flatten()), set_nan_inf(item[0][1].flatten()))) for i, item in tqdm(enumerate(dataset[batch_len*batch:(batch_len*batch+batch_len)]))]
    data = torch.stack(dataset, dim=0)
    print('Dataset array shape: ', data.shape)
    data = data.numpy()
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    #kmeans = KMeans(n_clusters=n_clusters,)
    #kmeans.fit(data)
    K = range(8, 20)
    fits = []
    score = []


    for k in tqdm(K):
        # train the model for current value of k on training data
        model = KMeans(n_clusters = k, random_state = 0, n_init='auto').fit(data)
        
        # append the model to fits
        fits.append(model)
        
        # Append the silhouette score to scores
        score.append(silhouette_score(data, model.labels_, metric='euclidean'))

    print(fits, score)
    ind = score.index(score.max())
    best_model = fits[ind]
    #best_model.labels_
    for i in range(len(dataset)):
        label = best_model.labels_[i]
        dataset[i] = label

def rescale_init(model, coeff):
    for name, param in model.named_parameters():
        param.data *= coeff

def save_torch(item, pth):
    torch.save(item, pth)

from torch.nn.utils import spectral_norm, remove_spectral_norm, weight_norm, remove_weight_norm

def apply_spectral_norm(module):
    """
    Recursively apply spectral normalization to all Linear and Conv layers in a module.
    """
    for name, child in module.named_children():
        if isinstance(child, (nn.Linear, nn.Conv2d)):
            setattr(module, name, spectral_norm(child))  # Replace layer with spectral normalized version
        else:
            apply_spectral_norm(child)

def remove_spectral_norm(module):
    """
    Recursively remove spectral normalization to all Linear and Conv layers in a module.
    """
    for name, child in module.named_children():
        if isinstance(child, (nn.Linear, nn.Conv2d)):
            remove_spectral_norm(child) # Replace layer with spectral normalized version
        else:
            remove_spectral_norm(child)

def apply_weight_norm_after_spectral_norm(module):
    """
    Remove spectral norm from all layers and apply weight norm instead.
    """
    for name, child in module.named_children():
        if isinstance(child, (torch.nn.Linear, torch.nn.Conv2d)):
            try:
                # Remove spectral norm if it was applied
                remove_spectral_norm(child)
            except AttributeError:
                pass  # Spectral norm not applied to this layer, skip

            # Apply weight norm
            setattr(module, name, weight_norm(child))
        else:
            apply_weight_norm_after_spectral_norm(child)

def apply_weight_norm(module):
    """
    Recursively apply weight normalization to all Linear and Conv layers in a module.
    """
    for name, child in module.named_children():
        if isinstance(child, (nn.Linear, nn.Conv2d)):
            setattr(module, name, weight_norm(child))  # Replace layer with spectral normalized version
        else:
            apply_weight_norm(child)

def remove_weight_norm(module):
    """
    Recursively remove spectral normalization to all Linear and Conv layers in a module.
    """
    for name, child in module.named_children():
        if isinstance(child, (nn.Linear, nn.Conv2d)):
            setattr(module, name, remove_weight_norm(child))  # Replace layer with spectral normalized version
        else:
            remove_spectral_norm(child)

def initialize_weights(m):
    init = nn.init.xavier_uniform_
    if isinstance(m, nn.Linear):
        init(m.weight) # Xavier initialization for linear layers
        if m.bias is not None:
            nn.init.zeros_(m.bias)         # Zero bias initialization
    elif isinstance(m, nn.MultiheadAttention):
        init(m.in_proj_weight)  # Initialize attention weights
        #m.in_proj_weight.data /= sqrt(243)
        nn.init.zeros_(m.in_proj_bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)           # Initialize LayerNorm weights to 1
        nn.init.zeros_(m.bias)  

def set_lr(optim, lr):
    for param_group in optim.param_groups:
        param_group['lr'] = lr

def linear_layer_lr_scaling(module_list, base_lr, max_lr, extra_params):
    """
    Create layer-wise learning rates that scale linearly across the layers in the ModuleList.
    
    Args:
        module_list (torch.nn.ModuleList): The layers of the network.
        base_lr (float): The learning rate for the first layer.
        max_lr (float): The learning rate for the last layer.
        
    Returns:
        list: A list of dictionaries with parameters and their corresponding learning rates.
    """
    num_layers = len(module_list)
    lr_configs = []

    for idx, layer in enumerate(module_list):
        # Linearly scale the learning rate for this layer
        layer_lr = base_lr + (max_lr - base_lr) * (idx / (num_layers - 1))
        
        # Add the layer parameters with the calculated learning rate to the list
        lr_configs.append({'params': layer.parameters(), 'lr': layer_lr})
    
    if extra_params is not None:
        lr_configs.append({
            'params': extra_params,
            'lr': base_lr,
        })

    return lr_configs
    
    return lr_configs

#@torch.compile
def generate_meta_model_dataset(dataset_pth, proportion, model_pth, device='cuda',
                                  dtype=torch.float32, num_bins=200):
    print('Generating Meta Model Dataset...')
    model = torch.load(model_pth).eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    dataset = pic_load(dataset_pth)
    losses = [None for i in range(len(dataset))]
    t_acts = [None for i in range(len(dataset))]
    preds = [None for i in range(len(dataset))]
    exp_price_diffs = [None for i in range(len(dataset))]
    new_dataset = []
    dataset = DataLoader(dataset)
    bin_edges = torch.load(f'Bin_Edges_{num_bins}')
    # Calculate the losses for the dataset
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataset)):
            target = None
            data = data[0]
            data, summary, prices = data[0], data[1], data[2]
            data = set_nan_inf(data)
            
            # For light data augmentation
            #data = gaussian_noise(data, 1e-3) # Unsqueeze to add batch dim
            #summary = gaussian_noise(summary, 1e-3)
            #print(data.shape,summary.shape)
            pred, t_act = model.forward_with_t_act(data.to(device, dtype=dtype), summary.to(device, dtype=dtype))
            preds[i] = pred
            t_acts[i] = t_act
            buy_price = prices[0]
            #print(prices)
            for sell_price in prices[1:]:
                profit = [sell_price/buy_price]
                if target == None:
                    target = [_convert_reward_to_one_hot_batch(item, bin_edges) for item in profit]
                    target = torch.stack(target)
                else:
                    _target = [_convert_reward_to_one_hot_batch(item, bin_edges) for item in profit]
                    _target = torch.stack(_target)
                    target = torch.cat((target, _target), dim=0)

            target = torch.permute(target, (1, 2, 0)) # b x 100 x 4
            ti = target.size(2)
            for j in range(ti):
                if j == 0:
                    loss = loss_fn(pred[:,:,j], target[:,:,j].to(device))
                else:
                    loss += loss_fn(pred[:,:,j], target[:,:,j].to(device))
            losses[i] = loss
            exp_price = get_expected_price(pred[:,:,0],bin_edges)
            target_price = prices[1]/buy_price
            exp_price_diffs[i] = exp_price.cpu()-target_price
            #if i > 100:
            #    break
    #print(losses[0])
    for i, data in tqdm(enumerate(dataset)):
        new_dataset.append((t_acts[i], preds[i], losses[i], exp_price_diffs[i]))

    save_pickle(new_dataset, f'MM_{dataset_pth}')

def train_meta_model(model, dataset_pths, batch_size, epochs, lr, wd):
    from utils import _get_expected_price
    optim = Lion(model.parameters(),lr=lr,weight_decay=wd)
    loss_fn = torch.nn.MSELoss()
    datasets = []
    bin_edges = torch.load('Bin_Edges_300')
    print('Loading Datasets...')
    for pth in tqdm(dataset_pths):
        datasets.append(pic_load(pth))
    for i in range(epochs):
        print(f'Epoch {i+1}')
        #datasets = []
        for dataset in datasets:
            #dataset = pic_load(pth)
            #print(dataset[100])
            #visualize_binned_distribution(dataset[0][1][0,:,2].cpu())
            dataloader = DataLoader(dataset,batch_size,shuffle=True)
            
            running_loss = 0
            for i, data in enumerate(dataloader):
                #data, label = data[0], data[1].to('cuda')
                t_act, pred, label, diff = data[0].to('cuda'),data[1].to('cuda'), data[2].to('cuda'), data[3].to('cuda')
                diff = torch.abs(diff).float()
                #price_pred = _get_expected_price(pred,bin_edges)
                mm_pred = model(t_act,pred)
                #label = torch.nn.functional.one_hot(label,2)
                loss = loss_fn(mm_pred, diff)
                running_loss += loss.item()
                loss.backward()
                optim.step()
                optim.zero_grad()
                if i%10 == 0:
                    print(f'Running Loss: {running_loss/(i+1):.4f}\r', end=' ')

    torch.save(model,'MetaModel')

def generate_meta_model_dataset_with_labels(dataset_pth, proportion, model_pth, device='cuda',
                                            dtype=torch.float32, num_bins=320):
    model = torch.load(model_pth).eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    dataset = pic_load(dataset_pth)
    losses = [None for _ in range(len(dataset))]
    t_acts = [None for _ in range(len(dataset))]
    preds = [None for _ in range(len(dataset))]
    #jac_norms = [None for _ in range(len(dataset))]

    # Calculate the losses for the dataset
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataset)):
            target = None
            data = data[0]
            data, summary, prices = data[0], data[1], data[2]
            data = set_nan_inf(data)

            pred, t_act = model.forward_with_t_act(data.to(device, dtype=dtype), summary.to(device, dtype=dtype))
            preds[i] = pred
            t_acts[i] = t_act
            #jac_norms[i] = get_jacobian_norm(model,data,summary)
            buy_price = prices[0]
            for sell_price in prices[1:]:
                profit = [sell_price / buy_price]
                if target is None:
                    target = [convert_reward_to_one_hot_batch(item, num_bins) for item in profit]
                else:
                    _target = [convert_reward_to_one_hot_batch(item, num_bins) for item in profit]
                    _target = torch.stack(_target)
                    target = torch.cat((target, _target), dim=0)

            target = torch.permute(target, (1, 2, 0))  # b x 100 x 4
            ti = target.size(2)
            for j in range(ti):
                if j == 0:
                    loss = loss_fn(pred[:, :, j], target[:, :, j].to(device))
                else:
                    loss += loss_fn(pred[:, :, j], target[:, :, j].to(device))

            losses[i] = loss.item()

    # Determine the threshold for the top proportion
    sorted_losses = sorted(losses)
    num_items_to_keep = int(len(dataset) * proportion)
    threshold = sorted_losses[num_items_to_keep - 1]  # Last loss in the top proportion

    # Generate new dataset with binary labels
    new_dataset = []
    for i, data in enumerate(dataset):
        is_top_proportion = losses[i] <= threshold  # Binary indicator
        # Add a new field/value to the dataset
        new_data = (t_act[i], pred[i], is_top_proportion)
        new_dataset.append(new_data)

    # Save the updated dataset
    save_pickle(new_dataset, f'{proportion}P_MM_{dataset_pth}')
    print(f"New dataset with binary labels saved to {proportion}P_Easy_WithLabels_{dataset_pth}")

def get_bin_width(dataset,num_bins):
    dataset = DataLoader(dataset)
    all_profits = []
    for i, item in tqdm(enumerate(dataset)):
        item = item[0]
        prices = item[2]
        buy_price = prices[0]
        for sell_price in prices[1:]:
            profit = sell_price/buy_price
            #if i == 0:
            #    print(profit)
            if profit < 0.1:
                profit = torch.tensor(0.5)
            if profit > 1.5:
                profit = torch.tensor(1.5)
            all_profits.append(profit.item())
        if i > 30000:
            break
    all_profits = np.array(all_profits)
    bin_edges = np.quantile(all_profits, q=np.linspace(0,1,num_bins+1))
    bin_edges = torch.tensor(bin_edges)
    print(bin_edges)
    save_torch(bin_edges,f'Bin_Edges_{num_bins}')

def main():
    seed = 155
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    seq_len = 600
    trade_period = 5
    k = 20
    c_prop = 0.25
    dataloader_pth_list = []
    dataloader_pth = f'S_Price_Dataset_[a-n, seq-{seq_len}, n-{trade_period}, k%{k}, c_prop-{c_prop}].pickle'
    o_dataloader_pths = ['0.25P_Oumodel = tlier_' + item for item in dataloader_pth_list]
    
    d_list = [item for item in os.listdir() if item.startswith('Dataset')]

    # If Training was interuppted, you can specify datasets that were already covered to ignore
    exclude_list = [1,2,3,5,7,8]
    d_list = [item for item in d_list if not int(item[-1]) in exclude_list]
    print(f'Effective Num Datasets: {len(d_list)}/14')

    dataloader_pth_list = d_list
    model_pth = 'DistPred_m_2'
    #model_pth = 'full_model_so_E8_E3'
    data_dict_pth = 'DataDict_2024-06-26'
    summaries_pth = 'a-z_summary_embs.pickle'
    sectors_pth = 'a-z_sector_dict'
    prices_pth = 'unnorm_price_series_a-z_-10y-2024-06-26__.pickle'
    iter = 3
    generate_data = 0
    genreate_layer_data = 0
    np_dataset = 0
    train_outlier = 0
    use_warmup = 1
    tr_meta_model = 0

    if tr_meta_model:
        num_bins=300
        generate_mm_data = 0
        if generate_mm_data:
            for item in d_list:
                #if not item.endswith('+10'):
                #   print(item)
                generate_meta_model_dataset(item,0.1,model_pth,num_bins=num_bins)
        w = 1000
        d = 4
        epochs = 20
        bs = 64
        lr = 3e-6
        wd = 0.03
        model = meta_model(w,d).to('cuda')
        mm_data_pths = [item for item in os.listdir() if item.startswith('MM')]
        print(f'Num Meta Model Datasets: {len(mm_data_pths)}')
        train_meta_model(model,mm_data_pths,bs,epochs,lr,wd)
        return 0

    #gauss_normalize_dataset(d_list[0])
    #if gauss_normal:
    #    for item in d_list:
    #        _sgauss_normalize_dataset(item)

    if genreate_layer_data:
        for item in dataloader_pth_list:
            generate_naive_layer_2_dataset(item, model_pth, half=0)
            generate_naive_layer_2_dataset(item, model_pth, half=1)
            #generate_naive_layer_dataset(item, model_pth,half=1)
    if generate_data: 
        for i in range(15):
            dataloader_pth = f'Dataset_[a-z, seq-600, n-5, c_prop-{c_prop}]_{i+1+iter}_k{k}'
            dataloader = QTrainingData(summaries_pth, prices_pth, seq_len, trade_period,
                                    k=k, c_prop=c_prop, full=True, load=1, 
                                    sectors_pth=sectors_pth, data_slice_num=i+iter,
                                    data_dict_pth=data_dict_pth)
            save_pickle(dataloader.dataloader, dataloader_pth)
    if np_dataset:
        for item in dataloader_pth_list:
            #generate_model_outlier_dataset(item, proportion=0.05, model_pth=model_pth)
            generate_outlier_dataset(item, 0.15)
    if train_outlier:
        dataloader_pth_list = o_dataloader_pths
    gen_bin_indices = 0
    if gen_bin_indices:
        print('Loading Dataset for bin indices generation...')
        dataset = pic_load(d_list[0])
        num_bins=300
        get_bin_width(dataset,num_bins)
    
    batch_size = 1
    lr = 1e-4
    num_bins = 300
    #dim = 52
    dim = 218
    ff_dim = 55000
    n_head = 1
    layers = 35
    epochs = 409
    scale = 0.07
    num_lin_layers = 3
    dtype = torch.float32
    t_max = 1892
    w_decay = 0.1
    grd_nrm = 1.0
    noise_level = 5e-7
    full_stack = 0
    grad_acc_steps = int(64/batch_size)
    init_scale = 0.7
    dropout = 0.2
    train_prop = 0.2


    debug_cuda = False
    train = 1
    gen_model = 1
    train_lora = 0
    k_means = 0

    if k_means:
        dataset_k_means(pic_load(d_list[0]), batch=0)

    if debug_cuda:
        os.environ["NCCL_P2P_LEVEL"] =  "NVL"
        os.environ["NCCL_SHM_DISABLE"] = '1'
        os.environ['NCCL_DEBUG'] = 'INFO'
        os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'
        #torch.cuda.empty_cache()
        print(f"CUDA_VISIBLE_DEVICES in Python: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        print(torch.cuda.device_count())
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['TORCH_USE_CUDA_DSA'] = '1'
        print("CUDA available:", torch.cuda.is_available())
        print("Current device:", torch.cuda.current_device())
        print("All available devices:", [torch.cuda.device(i) for i in range(torch.cuda.device_count())])
    
    if train:
        use_wandb = 0
        use_dist = 0
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        if use_dist:    
            local_rank = int(os.environ["LOCAL_RANK"])
            dist.init_process_group(backend='nccl')
            torch.multiprocessing.set_sharing_strategy('file_system')
            dataset = pic_load(dataloader_pth)
            sampler = DistributedSampler(dataset)
            dataloader = DataLoader(dataset, batch_size, shuffle=False, sampler=sampler)
            model = torch.load(model_pth).to(f'cuda:{local_rank}')
            #model = Models.GL_DirectPredSumTran(seq_len=seq_len, data_dim=dim, nhead=n_head, ff=ff_dim, layers=layers, scale=1).to(f'cuda:{local_rank}',dtype=dtype)
            model = DDP(model)
            model = torch.compile(model) 
        generate_stack=0
        if generate_stack:
            transfer_model = torch.load(model_pth)
            get_model_parameter_count(transfer_model)
            model = t_Dist_Pred(seq_len,dim,num_bins,nhead=n_head,ff=ff_dim,layers=layers,sum_emb=832,scale=scale,dropout=dropout)
            get_model_parameter_count(model)
            for i in range(28):
                model.layers[i] = transfer_model.layers[i]
            model = replace_square_linear_layers(model)
            get_model_parameter_count(model)
            for i in range(14):
                for param in model.layers[i].parameters():
                    param.requires_grad = False
            

        else:
            # = [dataloader_pth, dataloader_pth2]
            #dataloader = pic_load(dataloader_pth)
            #data_len = len(dataloader)
            #dataloader = DataLoader(dataloader, batch_size, pin_memory=True, shuffle=True)
            if not gen_model:
                model = torch.load(model_pth)
                load_optim = 1
            if full_stack and gen_model:
                l_model_pth = 'L1_Dist600_E40'
                l2_model_pth = 'mL2_Dist600_E16.5_E59'
                b_model_pth = 'Dist600__E12'
                model_pth = 'full_model_so'
                #model = Full_L1_Dist_Pred(b_model_pth, l_model_pth, train=True)
                model = Composed_Dist_Pred(b_model_pth, l_model_pth, l2_model_pth, train=True)
                #model = Full_Dist_Pred('full_model_E8')
                load_optim = 0
            elif full_stack:
                model = torch.load(model_pth)
                a = 0
                if a:
                    # Only train the base layer
                    #for pram in model.layer.parameters():
                    #    pram.requires_grad = True
                    freeze = True
                    #for param in model.layer2.linear_out.parameters():
                    #    param.requires_grad = freeze
                    for param in model.layer2.linear_in.parameters():
                        param.requires_grad = freeze
                    for param in model.layer2.cls_head_in.parameters():
                        param.requires_grad = freeze
                    for param in model.layer.linear_out.parameters():
                        param.requires_grad = freeze
                    for param in model.layer.linear_in.parameters():
                        param.requires_grad = freeze
                    for param in model.layer.cls_head_in.parameters():
                        param.requires_grad = freeze
                    for param in model.base.linear_in.parameters():
                        param.requires_grad = freeze
            elif gen_model:
                model = t_Dist_Pred(seq_len=seq_len, data_dim=dim, num_bins=num_bins, nhead=n_head, ff=ff_dim, layers=layers, sum_emb=832, scale=scale, dropout=dropout).to('cuda', dtype=dtype)
                #model.apply(initialize_weights)
                rescale_init(model, init_scale)
                model.linear_in = replace_square_linear_layers(model.linear_in,perms=3)
                #apply_weight_norm(model)
                model_pth = 'DistPred_m_2'
                load_optim = 0
            train_upper = 0
            if train_upper:
                for param in model.layer.parameters():
                    param.requires_grad = False
                for param in model.base.parameters():
                    param.requires_grad = False
            #torch.cuda.empty_cache()
            #model = L2_Dist_Pred(seq_len=seq_len, data_dim=dim*3, num_bins=num_bins, nhead=n_head, ff=ff_dim, layers=layers, sum_emb=832, scale=scale).to('cuda', dtype=dtype)
            #model = DataParallel(model).to('cuda')
            model = model.to('cuda')
            model.train()
        if use_wandb:
            wandb.init(
            project = 'Dist Prediction',
            config={
                'model_name' : model_pth,
                'lr' : lr,
                'ff_dim' : ff_dim,
                'Batch size' : batch_size,
                'layers' : layers,
                'dtype' : dtype,
                't_max' : t_max,
                'w_decay' : w_decay,
                'grd_nrm' : grd_nrm,
                'trade period' : trade_period,
                'nhead' : n_head,
                'scale' : scale,
                'dataset' : dataloader_pth_list
            }
            )
        #for name, param in model.named_parameters():
        #    if name.startswith('layers.0'):
        #        param.requires_grad=False
        #    elif name.startswith('layers'):
        #        param.requires_grad=True
        world_size = 2
        torch.set_float32_matmul_precision('high')
        save_rate = 1
        misc = [batch_size, trade_period, dataloader_pth_list, save_rate, noise_level, full_stack]
        params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad==True)
        print('Network size: ', params, '\nTrainable: ', trainable_params)
        #save_name = f'Dist_pred_{params}'
        save_name = f'{model_pth}_'
        load_optim = 0
        #model = torch.compile(model)
        model = model.to('cuda')
        #model.train()
        torch.compile(Train_Dist_Direct_Predictor)
        
        
            #print(name)
        torch.cuda.empty_cache()
        #remove_spectral_norm(model)
        #apply_weight_norm_after_spectral_norm(model)
        Train_Dist_Direct_Predictor_SAM(model, epochs, save_name, lr, num_bins, t_max, w_decay, grd_nrm, misc=misc, 
                                    device=f'cuda', dtype=dtype, load_optim=load_optim, grad_acc_steps=grad_acc_steps,
                                    use_warmup=use_warmup, train_prop=train_prop)
        #Train_Layer_Dist_Direct_Predictor(model, epochs, save_name, lr, num_bins, t_max, w_decay, grd_nrm, misc=misc, device=f'cuda', dtype=dtype)
    else:
        pass
    
    
if __name__ == '__main__':
    main()