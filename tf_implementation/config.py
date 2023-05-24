import datetime
import os
import json

class Config:
    def __init__(self,base_kw_args):
        self.update(base_kw_args)
        
    def update(self,overload_config):
        if overload_config == {}:
            print('updated with empty config!')
        for k, v in overload_config.items():
            try:
                setattr(self, k, v)
            except AttributeError:
                print(k,v)

    def load_version(self,name):
        print(f'version set to {name}')
        log_dir = name.replace('ckpts','logs')
        setting_dir = name.replace('ckpts','settings')
        self.train_log_dir = os.path.join(log_dir,
                                          'train')
        self.test_log_dir = os.path.join(log_dir,
                                         'test')

        self.best_ckpt_path = os.path.join(name,
                                          'best_models')
        self.last_ckpt_path = os.path.join(name,
                                          'last_model')

        self.update(json.load(open(setting_dir+'.json','r')))

        assert len(self._early_stopping_patience) == len(self.strategies),\
            f'number of strategies except ground truth {len(self.strategies)}\
            and early stopping patience {len(self._early_stopping_patience)} should be equal'

    def create_version(self,version_name):
        version_name += '_'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = self._log_dir+version_name
        ckpt_dir = self._ckpt_path+version_name

        self.train_log_dir = os.path.join(log_dir,
                                          'train')
        self.test_log_dir = os.path.join(log_dir,
                                         'test')

        self.best_ckpt_path = os.path.join(ckpt_dir,
                                          'best_models')
        self.last_ckpt_path = os.path.join(ckpt_dir,
                                          'last_model')

        os.mkdir(log_dir)
        os.mkdir(ckpt_dir)
        os.mkdir(self.train_log_dir)
        os.mkdir(self.test_log_dir)
        os.mkdir(self.best_ckpt_path)
        os.mkdir(self.last_ckpt_path)
        json.dump({},open(self._setting_path+version_name+'.json','w'))
        print('version created succesfully')
