import Models
import datetime
import torch
import torch.nn as nn
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from Stock import a_lot_of_stocks, stock_info, all_stocks
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import itertools
import wandb
# Two representations to encourage similarity are:
# 1. Autoencoded news title embeddings
# 2. Autoencoded stock price embeddings (movement over next few weeks)


#NewsVAE = Models.NewsEmbeddingVAE().to('cuda')
#d_enc = Models.mpNetEncoder().to('cuda')
#rob_enc = Models.RobertaEncoder().to('cuda:3')
articles_pic = 'articles_1.pickle'
full_data_pic = 'stock_data.pickle'
        

def GetDailyNewsEmbeddings(start, end, num_articles=100, save_name='mp_embeddings_ae', stocks=a_lot_of_stocks):
    start = datetime.date(start[0], start[1], start[2])
    end = datetime.date(end[0], end[1], end[2])
    #news = np.zeros(len(stocks))
    i = 0
    getter = stock_info(stocks)
    save_name = f'{save_name}.pickle'
    news = getter.get_news(start, end, num_articles)
    #for i in range(len(news)):
    #    news[i] = d_enc(news[i])
    with open(save_name, 'wb') as outfile:
            pickle.dump(news, outfile,)
    print(f"\nSaved to {save_name}")

def get_articles(data, pth=articles_pic):
    # Returns just the articles from full_data_series to 
    # Do unsupervised learning
    date_len = len(data.keys())
    num_companies = len(data.values()[0].keys())
    num_articles = len(data.values()[0][1].values()[0])
    out_data = []
    for date in data.keys():
        article_dict = data[date][1]
        for company in article_dict.keys():
            out_data.append(article_dict[company])
    out_data = np.array(out_data)
    out_data = np.hstack(out_data)
    for i in range(len(out_data)):
        out_data[i] = Models.DailyNewsEncoder(out_data[i])
    mean = np.mean(out_data)
    std = np.std(out_data, ddof=1)
    data_with_normalization = {0 : out_data, 1:(mean, std)}
    with open(pth, 'wb') as outfile:
        pickle.dump(data_with_normalization, outfile)
    
def pic_load(pic):
    with open(pic, 'rb') as infile:
        return pickle.load(infile)

class news_dataset(Dataset):
    def __init__(self, article_dict):
        a = list(article_dict.values())
        a = list(itertools.chain.from_iterable(a))
        self.article_arr = a

    def __len__(self):
        return len(self.article_arr)
    
    def __getitem__(self,index):
        return self.article_arr[index]

class price_dataset(Dataset):
    def __init__(self, data):
        self.article_arr = data

    def __len__(self):
        return len(self.article_arr)
    
    def __getitem__(self,index):
        return self.article_arr[index]
 
def get_company_summary_embs(data_dict):
    a = stock_info(data_dict, 'a-n')
    return a.get_company_text_info()

def ddp_setup(rank=3,world_size=4):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group(backend='nccl', rank=rank, world_size=world_size)

def embed_titles(encoder, data_name='article_titles_a_2022-06-01_2022-08-01_100.pickle'):
    with open(data_name, 'rb') as infile:
        article_dict = pickle.load(infile)
    del_items = []
    for ticker, item in article_dict.items():
        if len(item) == 0:
            print(ticker, item)
            del_items.append(ticker)
        else:
            #print(item)
            article_dict[ticker] = [encoder(title) for title in item]
    for item in del_items:
        del article_dict[item]
    save_name = f"roberta_embd_{data_name}"
    with open(save_name, 'wb') as out:
        pickle.dump(article_dict, out)
    
def noise_tensor(input, level, normalization=(0,1)):
    shape = input.shape
    noise = torch.randn(shape).to('cuda')
    noise = noise*level*normalization[1]+level*normalization[0]
    return noise

def train_NewsVAE(model, lr, news_dataloader, epochs, pth='NewsVAE.pth', noise_level=0.1):
    #ddp_setup()
    #print('Setup complete')
    #model = DDP(model)
    #print('Model Wrapped')
    model.train()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        running_loss = 0
        last_loss = 0
        print(f"\nEpoch: {epoch+1}")
        for i, emb in enumerate(news_dataloader):
            noise_emb = noise_tensor(emb, level=noise_level)
            pred = model(noise_emb)
            loss = loss_fn(pred, emb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()
            if i > 0 and i % 50 == 0:
                last_loss = running_loss/i
                print(f"\rBatch loss: {last_loss}", end='')
    #destroy_process_group()
    torch.save(model.state_dict(), pth)

def save_pickle(name, d):
        with open(name, 'wb') as out:
            pickle.dump(d, out)

def gauss_normalize(tensor, dim):
    a = (tensor-torch.mean(tensor, dim=dim, keepdim=True))/torch.std(tensor, dim=dim, unbiased=False, keepdim=True)
    return a.clone().detach()

class n_seq_stocks_dataset(Dataset):
    # n day lengths
    def __init__(self, data_pth, n, summaries_pth=None, name_str='a-j', price_only=True, k=2):
        self.n = n
        with open(data_pth, 'rb') as infile:
            self.p_data = pickle.load(infile)
        self.convert_dtinx_tensor()
        self.data_len = list(self.p_data.values())[0].shape[1]
        self._data = {}
        self.save_name = f'{n}_seq_{self.data_len}_{name_str}_stock_ae_dataset.pickle'
        if price_only:
            self.format_price_only()
        else:
            with open(summaries_pth, 'rb') as infile:
                self.summaries = pickle.load(infile)
            self.format(k)        

    def convert_dtinx_tensor(self):
        del_items = []
        for company, prices in self.p_data.items():
            try:
                d = torch.stack(list(prices.values()), dim=1)
                self.p_data[company] = d
            except Exception as e:
                del_items.append(company)
                
        for item in del_items:
            del self.p_data[item]
            
    def format_price_only(self):
        num_additions = self.data_len-self.n
        del_items = []
        for stock, data in self.p_data.items():
            #print(data.shape)
            self._data[stock] = []
            try: 
                for i in range(num_additions):
                    if data[i:self.n+i, :].shape == torch.Size([self.n, 5]):
                        self._data[stock].append(data[i:self.n+i, :])
                    #print(self._data.shape)
            except Exception as e:
                del_items.append(stock)
                print(e)
        for item in del_items:
            del self._data[item]
        a = list(self._data.values())
        a = list(itertools.chain.from_iterable(a))
        #print(a[0].shape)
        #print(len(a))
        #print(a[0], a[0][0].shape, a[0][1].shape, '\n',)
        #a = list(itertools.chain.from_iterable(a))
        self._data = a

    def format(self, k):
        num_additions = self.data_len-self.n
        del_items = []
        print
        for stock, data in self.p_data.items():
            self._data[stock] = []
            for i in range(num_additions):
                #print(data[i:self.n+i, :])
                if i % k == 0:
                    #print(data[:, i:self.n+i].shape[1], self.n)
                    try:
                        if data[:, i:self.n+i].shape[1] == self.n:
                            self._data[stock].append((self.summaries[stock], data[:, i:self.n+i]))
                    #        print(1)
                    except Exception as e:
                        pass
            #print(self._data[stock])
        for item in del_items:
            del self._data[item]
        a = list(self._data.values())
        a = list(itertools.chain.from_iterable(a))
        a = [(b[0], torch.transpose(b[1], 0, 1)) for b in a ]
        print(a[0], len(a))
        self._data = a
    
    def __getitem__(self, i):
        return self._data[i]
    
    def __len__(self):
        return len(self._data)

class market_seq_dataset(Dataset):
    def __init__(self, sum_pth, price_pth, n, k):
        # bulk prices -> company : date : data
        self.summaries = pic_load(sum_pth)
        self.bulk_prices = pic_load(price_pth)
        self.companies = list(self.summaries.keys())
        self.days = list(self.bulk_prices.keys())
    
    def get_market_day_data(self, date, n):
        data = {}
        for company in self.companies:
            f = True
            try:
                p = self.bulk_prices[company][date]
            except Exception as e:
                f = False
            if f:
                s = q

def combine_emb_price(emb, price):
    emb = emb.squeeze(1).unsqueeze(-1).unsqueeze(-1)
    emb = emb.expand(-1, -1, price.shape[1], price.shape[2])
    price = price.unsqueeze(1)
    return emb + price

def prepare_ae_data(data):
    emb = data[0].to('cuda', dtype=torch.float32)
    prices = data[1].to('cuda', dtype=torch.float32)
    if torch.isnan(prices).any():
        nan_mask = torch.isnan(prices)
        prices[nan_mask] = 0
    if torch.isinf(prices).any():
        inf_mask = torch.isinf(prices)
        prices[inf_mask] = 0
    prices = gauss_normalize(prices, dim=1)
    return emb, prices

def train_n_seq_stockVAE(model, dataloader, epochs, pth='PricesVAE.pth', lr=0.00000002, noise_level=0.00, grad_norm=1.0):
    model.train()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_data = len(dataloader)
    print(num_data)
    for epoch in range(epochs):
        a_loss = 0
        last_loss = 0
        running_loss = 0
        print(f"\nEpoch: {epoch+1}")
        i = 0
        #if epoch > 1 and noise_level > 0.04:
        #    noise_level -= 0.02
        for data in dataloader:
            i += 1
            emb = data[0].to('cuda', dtype=torch.float32)
            prices = data[1].to('cuda', dtype=torch.float32)
            prices = gauss_normalize(prices, dim=1)
            if torch.isnan(prices).any():
                nan_mask = torch.isnan(prices)
                prices[nan_mask] = 0
            if torch.isinf(prices).any():
                inf_mask = torch.isinf(prices)
                prices[inf_mask] = 0
            noise_prices = noise_tensor(prices, level=noise_level).to('cuda', dtype=torch.float32)
            p_pred, s_pred = model(noise_prices, emb)
            loss = 0.85*loss_fn(p_pred, prices)
            loss += 0.15*loss_fn(s_pred, emb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            a_loss += loss.item()
            running_loss += loss.item()
            if i > 0 and i % 50 == 0:
                last_loss = a_loss/50
                rl = running_loss/i
                print(f"\rBatch loss: {last_loss:.6f} | Running loss: {rl:.6f} | iters:{i}/{num_data}", end='')
                #if i % 20000 == 0:
                #    wandb.log({'loss':rl})
                a_loss = 0
    torch.save(model, pth)
    #wandb.finish()

def merge_data_dicts(p, s):
    s_tickers = list(s.keys())
    p_tickers = list(p.keys())
    tickers = set(s_tickers) & set(p_tickers)
    r_data = {}
    for ticker in tickers:
        r_data[ticker] = [p[ticker], s[ticker]]
    return r_data

def train_sum_VAE(model, data, epochs, pth, lr, noise_level):
    model.train().to('cuda')
    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    for epoch in range(epochs):
        print(f'Epoch: {epoch+1}')
        running_loss = 0
        c = 0
        print(type(data))
        for t, emb in data.items():   
            b = torch.mean(emb).to('cuda')
            v = torch.std(emb).to('cuda')
            _emb = noise_tensor(emb.to('cuda'), noise_level, (b, v))     
            pred = model(_emb.to('cuda'))
            loss = loss_fn(emb, pred)
            loss.backward()
            optim.step()
            optim.zero_grad()
            running_loss += loss.item()
            c += 1
            print(f'\rloss: {running_loss/c}', end='')
    torch.save(model, pth)

def encode_seq(model, data, name):
    model.eval().to('cuda')
    r_data = {}
    c = 0
    with torch.no_grad():
        for ticker, emb in data.items():
            r_data[ticker] = model.encode(emb)
            c += 1
            print(f'\r{c}/{len(data)}')
    save_pickle(name, r_data)

def main():
    prices = 'unnorm_price_series_a-n_-5y-2024-04-29__.pickle'
    summaries = 'a-n_ae_summary_embs.pickle'
    #a = n_seq_stocks_dataset(prices, 100, summaries, price_only=False, k=3)
    #b = DataLoader(a, 32, True)
    dae = 'a-n_sum+price_DAE_Dataloader_100_2k.pickle'
    #save_pickle(dae, b)
    dataloader = pic_load(dae)
    batch_size = 32
    lr = 1e-6
    epochs = 9
    noise_level = 0.00
    grad_norm = 1
    model = Models.MixedLayeredDAE(100).to('cuda')
    #model = torch.load('PricesSumVAE2.pth')
    pth = 'MixedDAE_250_L1.pth'
    print('Model Size:', sum(param.numel() for param in model.parameters()))
    train_n_seq_stockVAE(model, dataloader, epochs, pth, lr=lr, noise_level=noise_level, grad_norm=grad_norm)

if __name__ == '__main__':
    main()


#files = ['article_titles_a_2022-06-01_2022-08-01_100.pickle', 'mp_embeddings_a_2021-06-01_2022-08-01_1000.pickle']