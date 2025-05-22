import pandas as pd
import argparse
from trainer import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=int)
    parser.add_argument('-d', type=str)
    parser.add_argument('-seed_path', type=str)
    parser.add_argument('-class_samples_seed', type=int)
    parser.add_argument('-shot', type=str)
    a=parser.parse_args()
    exps=pd.read_csv('./args/'+a.d+'_publish.csv')
    args=exps[exps['ID']==a.i].to_dict('records')[0]
    args['seed']=[args['seed']]
    args['device']=[args['device']]
    args['do_not_save']=False
    args['seed_path']=a.seed_path
    args['class_samples_seed']=a.class_samples_seed
    args['shot']=a.shot
    # breakpoint()
    _, avg_acc = train(args)
    print(avg_acc)
    return avg_acc

if __name__ == '__main__':
    main()