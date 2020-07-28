from functions import *
from models import *
# import plotly.graph_objects as go
# import plotly.figure_factory as ff
# from Bio.SeqUtils import GC
# import pickle
import utils
from datetime import datetime

#CONSTANTS AND HYPERPARAMETERS (add to yaml)
# Device configuration
def run(config):
    time = datetime.now()
    time = time.strftime("%m-%d-%Y-%H:%M:%S")
    state_dict = {'itr': 0, 'epoch': 0, 'best_epoch': 0, 'best_test_loss': 9999999,
                  'config': config}

    if config['exp_name'] == '' and not config['resume']:
        config['exp_name'] = '{}_bs_{}_lr_{}'.format(time, config['batch_size'], config['lr'])
    elif (not len(config['exp_name']) == 0) and config['resume']:
        pass
    else:
        raise Exception('Set up experiment wrong!')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Cuda is available: {}'.format(torch.cuda.is_available()))
    # Hyper parameters
    #num_epochs = 10
    #batch_size = 100
    #learning_rate = 0.003
    #torch.backends.cudnn.benchmark = True
    print('torch.backends.cudnn.benchmark?? {}'.format(torch.backends.cudnn.benchmark))


    #load the dataset
    dataloaders, target_labels, train_out = get_data_loader(config,)

    #decode labels
    target_labels = [i.decode("utf-8") for i in target_labels]

    num_classes = len(target_labels) #number of classes

    #initialize the model
    model = ConvNetDeepCrossSpecies(config).to(device)

    criterion = nn.BCEWithLogitsLoss() #- no weights

    # optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    if config['resume']:
        print('Loading weights...')
        utils.load_weights(config, model, state_dict,)

    # code to train the model
    model, train_error, test_error = train_model(config, dataloaders['train'], dataloaders['valid'],
                                                 model, device, criterion, state_dict,
                                                 verbose=True)

    labels_E, outputs_E, labels_argmax, outputs_argmax = run_test(model, dataloaders['test'], device)

    compute_metrics(config, labels_E, outputs_E, labels_argmax, outputs_argmax)



def main():
    parser = utils.prepare_parser()
    config = vars(parser.parse_args())
    print(config)
    run(config)

if __name__ == '__main__':
  main()