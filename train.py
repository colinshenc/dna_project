from functions import *
from models import *
from model_DanQ import DanQ
# import plotly.graph_objects as go
# import plotly.figure_factory as ff
# from Bio.SeqUtils import GC
# import pickle
import json
import utils
from datetime import datetime

#CONSTANTS AND HYPERPARAMETERS (add to yaml)
# Device configuration
def run(config, plot_dict):
    time = datetime.now()
    time = time.strftime("%m-%d-%Y-%H:%M:%S")
    state_dict = {'itr': 0, 'epoch': 0, 'best_epoch': 0, 'best_test_loss': 9999999,
                  'config': config}

    if config['exp_name'] == '' and not config['resume']:
        config['exp_name'] = '{}_model_{}_fm_{}_bs_{}_lr_{}'.format(time, config['model'], config['feature_multiplier'], config['batch_size'], config['lr'])
    elif (not len(config['exp_name']) == 0):
        pass
    else:
        print(config['exp_name'])
        print(config['resume'])
        raise Exception('Set up experiment wrong!')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Cuda is available: {}'.format(torch.cuda.is_available()))
    # Hyper parameters
    #num_epochs = 10
    #batch_size = 100
    #learning_rate = 0.003
    torch.backends.cudnn.benchmark = True
    print('torch.backends.cudnn.benchmark?? {}'.format(torch.backends.cudnn.benchmark))


    #load the dataset
    #dataloaders, target_labels, train_out = get_data_loader(config,)
    dataloaders = get_data_loader(config,)

    #decode labels
    #target_labels = [i.decode("utf-8") for i in target_labels]

    #num_classes = len(target_labels) #number of classes

    #initialize the model
    if config['model'] == 'ours':
        model = ConvNetDeepCrossSpecies(config).to(device)
        criterion = nn.BCEWithLogitsLoss()  # - no weights
        # scheduler = lambda optimzer, _, _: optimizer # pass through.
        print('\n======>Our model<======\n')

    elif config['model'] == 'danq':
        '''their stuff...'''
        torch.manual_seed(1337)
        np.random.seed(1337)
        torch.cuda.manual_seed(1337)
        model = DanQ(config).to(device)
        criterion = nn.BCEWithLogitsLoss()  # Same as ours...

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=1)
        print('\n======>DanQ model<======\n')
    else:
        raise Exception('Choose one of the above model..')

    # criterion = nn.BCEWithLogitsLoss() #- no weights

    # optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    if config['resume']:
        with open("{}{}_results_log.txt".format(config['ckpts_path'], config['exp_name']), "a+") as file:
            file.write('\n\n\n------>Loading weights, resuming exp...\n')
        print('------>Loading weights, resuming exp...')
        utils.load_weights(config, model, state_dict,)

    # code to train the model
    model, train_error, test_error = train_model(config, dataloaders['train'], dataloaders['valid'],
                                                 model, device, criterion, state_dict,
                                                 verbose=True)

    labels_E, outputs_E, labels_argmax, outputs_argmax = run_test(model, dataloaders['test'], device)

    plot_dict = compute_metrics(config, labels_E, outputs_E, labels_argmax, outputs_argmax, plot_dict)

    return plot_dict
def main():

    parser = utils.prepare_parser()
    config = vars(parser.parse_args())
    if config['model'] =='ours':
        for _ in range(3):
            plot_dict={}
            plot_dict['roc_auc'] = []
            plot_dict['auprc'] = []
            for feat_mult in [8,16,48,96,128,192,256,300,320]:
                config['feature_multiplier'] = feat_mult
                with open("{}{}_results_log.txt".format(config['ckpts_path'], config['exp_name']), "a+") as file:
                    file.write('\n\n\n\n\n\n\n\n======>>>>>>{}\n\n'.format(config))
                    file.write('With Maxpool1d\n')
                print(config)
                print('\n\n')
                plot_dict = run(config, plot_dict)
            with open('{}{}_{}_with_Maxpool_data_for_plot.txt'.format(config['ckpts_path'], config['exp_name'], _), 'a+') as file:
                file.write(json.dumps(plot_dict))
                file.write('\n\n\n')

            #json.dump(plot_dict, open('{}{}_{}_json_data_for_plot.txt'.format(config['ckpts_path'], config['exp_name'], _), 'a+'))
    elif config['model'] == 'danq':
        for _ in range(3):
            plot_dict = {}
            plot_dict['roc_auc'] = []
            plot_dict['auprc'] = []
            # for feat_mult in [8, 16, 48, 96, 128, 192, 256, 300, 320]:
            #     config['feature_multiplier'] = feat_mult
            with open("{}{}_results_log.txt".format(config['ckpts_path'], config['exp_name']), "a+") as file:
                file.write('\n\n\n\n======>>>>>>{}\n\n'.format(config))
                file.write('With danq\n')
            print('\n\n')
            print(config)
            plot_dict = run(config, plot_dict)
            with open('{}{}_{}_data_for_plot.txt'.format(config['ckpts_path'], config['exp_name'], _),
                      'a+') as file:
                file.write(json.dumps(plot_dict))
                file.write('\n\n\n')
    else:
        raise Exception('choose the correct model..')
if __name__ == '__main__':
    main()
    # utils.plot()