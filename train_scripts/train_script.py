import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.handlers import ModelCheckpoint
from random import randint

from utils.ranking import *
from utils.log import console_log, comet_log
from utils.accuracy import RankAccuracy

def train(device, net, dataloader, val_loader, args, logger, experiment):
    def update(engine, data):
        # Load training sample
        input_left, input_right, label, left_original = data['left_image'], data['right_image'], data['winner'], data['left_image_original']
        input_left, input_right, label = input_left.to(device), input_right.to(device), label.to(device)
        attribute = data['attribute'].to(device)
        label = label.float()
        optimizer.zero_grad()

        # Forward the training sample
        forward_dict = net(input_left,input_right)
        output_rank_left, output_rank_right =  forward_dict['left']['output'], forward_dict['right']['output']

        if args.attribute == 'all':
            loss = compute_multiple_ranking_loss(output_rank_left, output_rank_right, label, criterion, attribute)
        else:
            loss = compute_ranking_loss(output_rank_left, output_rank_right, label, criterion)

        # Backward step
        loss.backward()

        optimizer.step()
        if scheduler:
            scheduler.step()

        return  { 'loss':loss.item(),
                'rank_left': output_rank_left,
                'rank_right': output_rank_right,
                'label': label
                }

    def inference(engine,data):
        with torch.no_grad():
            # Load training sample
            input_left, input_right, label, left_original = data['left_image'], data['right_image'], data['winner'], data['left_image_original']
            input_left, input_right, label = input_left.to(device), input_right.to(device), label.to(device)
            attribute = data['attribute'].to(device)
            label = label.float()

            # Forward the training sample
            forward_dict = net(input_left,input_right)
            output_rank_left, output_rank_right =  forward_dict['left']['output'], forward_dict['right']['output']
            if args.attribute == 'all':
                loss = compute_multiple_ranking_loss(output_rank_left, output_rank_right, label, criterion, attribute)
            else:
                loss = compute_ranking_loss(output_rank_left, output_rank_right, label, criterion)

            return  { 'loss':loss.item(),
                'rank_left': output_rank_left,
                'rank_right': output_rank_right,
                'label': label
                }

    # Define model, loss, optimizer and scheduler
    net = net.to(device)
    
    class CustomJointLoss(nn.Module):
        def __init__(self, margin=0.2, weight=None, size_average=None, reduce=None, lam = 10):
            super(CustomJointLoss, self).__init__()
            self.lam = lam
            # self.binary_loss = nn.CrossEntropyLoss(weight=weight, size_average=size_average, reduce=reduce, reduction='mean')
            self.ranking_loss = nn.MarginRankingLoss(reduction='none', margin=margin)

        def forward(self, output1, output2, label):
            # compute cross-entropy loss
            # loss1 = self.binary_loss(output1, (label+1)/2)
            # loss2 = self.binary_loss(output2, 1-(label+1)/2)
            # binary_loss = loss1 + loss2

            # compute margin ranking loss
            ranking_loss = torch.mean(self.ranking_loss(output1, output2, label)**2)
            # combine the losses
            # loss = binary_loss/self.lam + ranking_loss
            loss = ranking_loss
            return loss
    
    # # function version (in case the class version not working, currently buggy)
    # L_b = nn.CrossEntropyLoss(weight=None, size_average=None, reduce=None, reduction='mean')
    # L_r = nn.MarginRankingLoss(reduction='none', margin=0.2)

    # def CustomJointLoss(output1, output2, label, binary_loss = L_b, ranking_loss = L_r, lam = 100):
    #     loss1 = binary_loss(output1, (label+1)/2)
    #     loss2 = binary_loss(output2, (label+1)/2)
    #     binary_loss = loss1 + loss2
    #     print(binary_loss)
    #     # compute margin ranking loss
    #     ranking_loss = torch.mean(ranking_loss(output1, output2, label)**2)
    #     print(ranking_loss)
    #     # combine the losses
    #     loss = binary_loss + lam * ranking_loss
    #     print(loss)
    #     return loss
    
    criterion = CustomJointLoss(margin = 0.2, lam = args.lam)



    optimizer = optim.Adam(net.parameters(), lr= args.lr, weight_decay=0.0, betas=(0.9, 0.98), eps=1e-09)
    if args.lr_decay:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.9, last_epoch=-1)
    else:
        scheduler = None

    # Engine specific parameters
    trainer = Engine(update)
    evaluator = Engine(inference)

    RunningAverage(output_transform=lambda x: x['loss'], device=device).attach(trainer, 'loss')
    RankAccuracy(output_transform=lambda x: (x['rank_left'], x['rank_right'], x['label']), device=device).attach(trainer,'acc')

    RunningAverage(output_transform=lambda x: x['loss'], device=device).attach(evaluator, 'loss')
    RankAccuracy(output_transform=lambda x: (x['rank_left'], x['rank_right'], x['label']),device=device).attach(evaluator,'acc')

    # Log training parameters after every epoch
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        net.eval()
        evaluator.run(val_loader)
        trainer.state.metrics['val_acc'] = evaluator.state.metrics['acc']
        net.train()
        if hasattr(net,'partial_eval'): net.partial_eval()
        metrics = {
                'train_loss':trainer.state.metrics['loss'],
                'acc': trainer.state.metrics['acc'],
                'val_acc': evaluator.state.metrics['acc'],
                'val_loss':evaluator.state.metrics['loss']
            }
        comet_log(
            metrics,
            experiment,
            epoch=trainer.state.epoch,
            step=trainer.state.epoch,
        )
        console_log(metrics,{},trainer.state.epoch)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_results(trainer):
        # Log training every 100th iteration
        if trainer.state.iteration %100 == 0:
            metrics = {
                    'train_loss':trainer.state.metrics['loss']
                }
            comet_log(
                metrics,
                experiment,
                step=trainer.state.iteration,
                epoch=trainer.state.epoch
            )
            console_log(
                metrics,
                {},
                trainer.state.epoch,
                step=trainer.state.iteration,
            )

    model_name = '{}_{}_{}_{}hidden'.format(args.model, args.premodel, args.attribute, args.hidden_layer)
    handler = ModelCheckpoint(args.model_dir, model_name,
                                n_saved=1,
                                create_dir=True,
                                save_as_state_dict=True,
                                require_empty=False,
                                score_function=lambda engine: engine.state.metrics['val_acc'])
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, {
                'model': net
                })

    if args.resume:
        def start_epoch(engine):
            engine.state.epoch = args.epoch
        trainer.add_event_handler(Events.STARTED, start_epoch)
        evaluator.add_event_handler(Events.STARTED, start_epoch)

    trainer.run(dataloader,max_epochs=args.max_epochs)

if __name__ == '__main__':
    from nets.MyCnn import MyCnn
    import torchvision.models as models

    net = MyCnn(models.resnet50)
    x = torch.randn([3, 244, 244]).unsqueeze(0)
    y = torch.randn([3, 244, 244]).unsqueeze(0)
    fwd = net(x, y)
    print(fwd)
