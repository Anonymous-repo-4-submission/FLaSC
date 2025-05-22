import copy
import logging
import numpy as np
import os
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from collections import defaultdict
from inc_net import ResNetCosineIncrementalNet,SimpleVitNet
from utils.toolkit import target2onehot, tensor2numpy, accuracy

num_workers = 8

class BaseLearner(object):
    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._classes_seen_so_far = 0
        self.class_increments=[]
        self._network = None

        self._device = args["device"][0]
        self._multiple_gpus = args["device"]
        self._weights = []
        
    def _compute_classwise_accuracy(self, y_pred, y_true):
        """Computes accuracy for each class"""
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        
        for pred, true in zip(y_pred, y_true):
            if pred == true:
                class_correct[true] += 1
            class_total[true] += 1

        class_accuracy = {
            class_id: (class_correct[class_id] / class_total[class_id]) * 100 
            if class_total[class_id] > 0 else 0
            for class_id in class_total
        }
        
        return class_accuracy
    
    def eval_task(self):
        y_pred, y_true, features = self._eval_cnn(self.test_loader)
        # class_accuracies = self._compute_classwise_accuracy(y_pred, y_true)
        
        #  # Print class-wise accuracy
        # print("Class-wise Accuracy:")
        # for class_id, acc in sorted(class_accuracies.items()):
        #     print(f"Class {class_id}: {acc:.2f}%")
       
        acc_total,grouped = self._evaluate(y_pred, y_true)
        return acc_total,grouped,y_pred[:,0],y_true,features
    
    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        features = []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                output = self._network(inputs)
                outputs = output["logits"]
                feature = output["features"]
            predicts = torch.topk(outputs, k=1, dim=1, largest=True, sorted=True)[1] 
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
            features.append(feature.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true), np.concatenate(features)

    # def _eval_cnn(self, loader):
    #     self._network.eval()
    #     # breakpoint()
    #     y_pred, y_true = [], []
    #     for _, (_, inputs, targets) in enumerate(loader):
    #         inputs = inputs.to(self._device)
    #         with torch.no_grad():
    #             # breakpoint()
    #             # outputs = self._network(inputs)["logits"]
                
    #             outputs = self._network(inputs)
    #             features = outputs['features']
    #             logits = outputs['logits']
    #             bs = features.shape[0]
    #             C = self._weights[0].shape[0]
    #             aggregated_output = torch.zeros(bs, C).to(self._device)
    #             for weight in self._weights:
    #                          # Perform matrix multiplication: features [bs, 768] @ weight.T [768, C] -> output [bs, C]
    #                 out = torch.matmul(features, weight.T)
    #                 aggregated_output += out
    #                 # breakpoint()
                
    #         # breakpoint()
    #         predicts = torch.topk(aggregated_output, k=1, dim=1, largest=True, sorted=True)[1] 
    #         y_pred.append(predicts.cpu().numpy())
    #         y_true.append(targets.cpu().numpy())
    #     return np.concatenate(y_pred), np.concatenate(y_true)  
    
    def _evaluate(self, y_pred, y_true):
        ret = {}
        acc_total,grouped = accuracy(y_pred.T[0], y_true, self._known_classes,self.class_increments)
        return acc_total,grouped 
    
    def _compute_accuracy_old(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)
    
    def _compute_accuracy(self, model, loader):
        # breakpoint()
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        if args["model_name"]!='ncm':
            if args["model_name"]=='adapter' and '_adapter' not in args["convnet_type"]:
                raise NotImplementedError('Adapter requires Adapter backbone')
            if args["model_name"]=='ssf' and '_ssf' not in args["convnet_type"]:
                raise NotImplementedError('SSF requires SSF backbone')
            if args["model_name"]=='vpt' and '_vpt' not in args["convnet_type"]:
                raise NotImplementedError('VPT requires VPT backbone')

            if 'resnet' in args['convnet_type']:
                self._network = ResNetCosineIncrementalNet(args, True)
                self._batch_size=128
            else:
                self._network = SimpleVitNet(args, True)
                self._batch_size= args["batch_size"]
            
            self.weight_decay=args["weight_decay"] if args["weight_decay"] is not None else 0.0005
            self.min_lr=args['min_lr'] if args['min_lr'] is not None else 1e-8
        else:
            self._network = SimpleVitNet(args, True)
            self._batch_size= args["batch_size"]
        self.args=args
        self.opt_lambda = -np.inf

    def after_task(self):
        self._known_classes = self._classes_seen_so_far
    
    def replace_fc(self,trainloader):
        self._network = self._network.eval()

        if self.args['use_RP']:
            #these lines are needed because the CosineLinear head gets deleted between streams and replaced by one with more classes (for CIL)
            self._network.fc.use_RP=True
            if self.args['M']>0:
                self._network.fc.W_rand=self.W_rand
            else:
                self._network.fc.W_rand=None

        Features_f = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_,data,label)=batch
                data=data.cuda()
                label=label.cuda()
                embedding = self._network.convnet(data)
                Features_f.append(embedding.cpu())
                label_list.append(label.cpu())
        Features_f = torch.cat(Features_f, dim=0)
        label_list = torch.cat(label_list, dim=0)
        
        Y=target2onehot(label_list,self.total_classnum)
        if self.args['use_RP']:
            #print('Number of pre-trained feature dimensions = ',Features_f.shape[-1])
            if self.args['M']>0:
                # Features_h=torch.nn.functional.relu(Features_f@ self._network.fc.W_rand.cpu())
                Features_h=Features_f@ self._network.fc.W_rand.cpu()

            
            else:
                Features_h=Features_f
            
            self.Q=self.Q+Features_h.T @ Y 
            # if self._cur_task == 0:
            self.G=self.G+Features_h.T @ Features_h

            if self.args['meta']:
                ridge=self.optimise_ridge_parameter_controlled_gmax(Features_h,Y)

            else:
                ridge=self.optimise_ridge_parameter(Features_h,Y)
                
            # breakpoint()
            
            
            Wo=torch.linalg.solve(self.G+ridge*torch.eye(self.G.size(dim=0)),self.Q).T #better nmerical stability than .inv
                
            self._network.fc.weight.data=Wo[0:self._network.fc.weight.shape[0],:].to(self._device)

        else:
            for class_index in np.unique(self.train_dataset.labels):
                data_index=(label_list==class_index).nonzero().squeeze(-1)
                if self.is_dil:
                    class_prototype=Features_f[data_index].sum(0)
                    # self._network.fc.weight.data[class_index]+=class_prototype.to(device='cuda') #for dil, we update all classes in all tasks
                    self._network.fc.weight.data[class_index]+=class_prototype.to(self._device)
                else:
                    #original cosine similarity approach of Zhou et al (2023)
                    class_prototype=Features_f[data_index].mean(0)
                    self._network.fc.weight.data[class_index]=class_prototype #for cil, only new classes get updated

    def optimise_ridge_parameter_controlled(self,Features,Y):
        ridges=10.0**np.arange(-8,9)
        num_val_samples=int(Features.shape[0]*0.8)
        losses=[]
        Q_val=Features[0:num_val_samples,:].T @ Y[0:num_val_samples,:]
        G_val=Features[0:num_val_samples,:].T @ Features[0:num_val_samples,:]
        for ridge in ridges:
            Wo=torch.linalg.solve(G_val+ridge*torch.eye(G_val.size(dim=0)),Q_val).T #better nmerical stability than .inv
            Y_train_pred=Features[num_val_samples::,:]@Wo.T
            losses.append(F.mse_loss(Y_train_pred,Y[num_val_samples::,:]))
        ridge=ridges[np.argmin(np.array(losses))]
        logging.info("Optimal lambda: "+str(ridge))
        if ridge > self.opt_lambda:
            self.opt_lambda = ridge
             
        logging.info("Optimal lambda: "+str(self.opt_lambda))   
        return self.opt_lambda
    
    def optimise_ridge_parameter_controlled_onestep(self, Features, Y):
        ridges = 10.0 ** np.arange(-8, 9)  # Search space for λ
        num_val_samples = int(Features.shape[0] * 0.8)  # 80% for training, 20% for validation
        losses = []

        # Convert to torch.Tensor if needed
        Features = torch.tensor(Features, dtype=torch.float32) if not isinstance(Features, torch.Tensor) else Features
        Y = torch.tensor(Y, dtype=torch.float32) if not isinstance(Y, torch.Tensor) else Y

        # Compute validation matrices
        Q_val = Features[:num_val_samples, :].T @ Y[:num_val_samples, :]
        G_val = Features[:num_val_samples, :].T @ Features[:num_val_samples, :]
        
        # Iterate over different ridge values
        for ridge in ridges:
            Wo = torch.linalg.solve(G_val + ridge * torch.eye(G_val.size(0), device=G_val.device), Q_val).T
            Y_train_pred = Features[num_val_samples:, :] @ Wo.T
            losses.append(F.mse_loss(Y_train_pred, Y[num_val_samples:, :]).item())

        # Select optimal lambda based on minimum loss
        optimal_lambda = ridges[np.argmin(losses)]
        logging.info(f"Optimal lambda: {optimal_lambda}")

        # Ensure opt_lambda is initialized
        current_lambda = getattr(self, "opt_lambda", None)

        if current_lambda is not None and current_lambda in ridges:
            current_idx = np.where(ridges == current_lambda)[0][0]
            optimal_idx = np.where(ridges == optimal_lambda)[0][0]

            # Update lambda only if the new lambda is one step away (±1 index)
            if abs(optimal_idx - current_idx) == 1:
                self.opt_lambda = optimal_lambda
        else:
            # If opt_lambda was not initialized, set it directly
            self.opt_lambda = optimal_lambda
            
        return self.opt_lambda
            
    
    def optimise_ridge_parameter_controlled_gmax(self, Features, Y):
        ridges = 10.0 ** np.arange(-8, 9)  # Search space for λ
        num_val_samples = int(Features.shape[0] * 0.8)  # 80% for training, 20% for validation
        losses = []

        # Convert to torch.Tensor if needed
        # Features = torch.tensor(Features, dtype=torch.float32) if not isinstance(Features, torch.Tensor) else Features
        # Y = torch.tensor(Y, dtype=torch.float32) if not isinstance(Y, torch.Tensor) else Y

        # Compute validation matrices
        Q_val = Features[:num_val_samples, :].T @ Y[:num_val_samples, :]
        G_val = Features[:num_val_samples, :].T @ Features[:num_val_samples, :]
        G_max = G_val.max().item()  # Maximum value in G_val

        # Determine the λ selection strategy based on shot setting
        shot =int(self.args["shot"])
        shot_threshold = 15  # Define a threshold between low and high-shot settings
        # breakpoint()
        if shot < shot_threshold:  # Low-shot setting
            lambda_condition = lambda lam: G_max / lam < 1  # Ensure Gmax * λ ≪ 1
        else:  # High-shot setting
            lambda_condition = lambda lam: G_max / lam > 1  # Ensure Gmax * λ ≫ 1

        # Iterate over different ridge values
        for ridge in ridges:
            Wo = torch.linalg.solve(G_val + ridge * torch.eye(G_val.size(0), device=G_val.device), Q_val).T
            Y_train_pred = Features[num_val_samples:, :] @ Wo.T
            losses.append(F.mse_loss(Y_train_pred, Y[num_val_samples:, :]))
            
        ridge=ridges[np.argmin(np.array(losses))]
            
            # Apply condition-based selection
            # if lambda_condition(ridge):
                # losses.append((loss, ridge))
        if self.opt_lambda == -np.inf:
            self.opt_lambda = ridge

        # breakpoint()
        # Select λ that minimizes loss while satisfying condition
        # if losses:
        #     optimal_lambda = min(losses, key=lambda x: x[0])[1]
        # else:
        #     optimal_lambda = ridges[np.argmin(losses)]  # Fallback: pick best λ

        logging.info(f"Optimal lambda for shot {shot}: {self.opt_lambda}")

        # # Update self.opt_lambda
        # if optimal_lambda > getattr(self, "opt_lambda", -np.inf):
        #     self.opt_lambda = optimal_lambda
        return self.opt_lambda
        
    def optimise_ridge_parameter(self,Features,Y):
        ridges=10.0**np.arange(-8,9)
        num_val_samples=int(Features.shape[0]*0.8)
        losses=[]
        Q_val=Features[0:num_val_samples,:].T @ Y[0:num_val_samples,:]
        G_val=Features[0:num_val_samples,:].T @ Features[0:num_val_samples,:]
        for ridge in ridges:
            Wo=torch.linalg.solve(G_val+ridge*torch.eye(G_val.size(dim=0)),Q_val).T #better nmerical stability than .inv
            Y_train_pred=Features[num_val_samples::,:]@Wo.T
            losses.append(F.mse_loss(Y_train_pred,Y[num_val_samples::,:]))
        ridge=ridges[np.argmin(np.array(losses))]
        logging.info("Optimal lambda: "+str(ridge))
        return ridge
    
    def incremental_train(self, data_manager):
        self.total_classnum = data_manager.get_total_classnum()
        self._cur_task += 1
        self._classes_seen_so_far = self._known_classes + data_manager.get_task_size(self._cur_task)
        if self.args['use_RP']:
            #temporarily remove RP weights
            del self._network.fc
            self._network.fc=None
        self._network.update_fc(self._classes_seen_so_far) #creates a new head with a new number of classes (if CIL)
        if self.is_dil == False:
            logging.info("Starting CIL Task {}".format(self._cur_task+1))
        logging.info("Learning on classes {}-{}".format(self._known_classes, self._classes_seen_so_far-1))
        self.class_increments.append([self._known_classes, self._classes_seen_so_far-1])
        self.train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._classes_seen_so_far),source="train", mode="train", meta_loader=self.args['meta'])
        self.train_loader = DataLoader(self.train_dataset, batch_size=self._batch_size, shuffle=False, num_workers=num_workers)
        train_dataset_for_CPs = data_manager.get_dataset(np.arange(self._known_classes, self._classes_seen_so_far),source="train", mode="test", meta_loader=False)
        self.train_loader_for_CPs = DataLoader(train_dataset_for_CPs, batch_size=self._batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._classes_seen_so_far), source="test", mode="test" , meta_loader=False)
        self.test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=num_workers)
        
        self.train_dataset_aug = data_manager.get_dataset(np.arange(self._known_classes, self._classes_seen_so_far),source="train", mode="train",meta_loader=self.args['meta'] )
  
        self.train_loader_aug = DataLoader(self.train_dataset_aug, batch_size=self._batch_size, shuffle=False, num_workers=num_workers)
        # self.train_dataset_aug =None
       
        logging.info(f"len_supervised data : {len(self.train_dataset)}")
        logging.info(f"len supervised data for G inc calculation : {len(train_dataset_for_CPs)}")
        logging.info(f"len_test data : {len(test_dataset)}")
        logging.info(f"classes learning unique, count: {np.unique(self.train_dataset.labels, return_counts=True)}")

        # breakpoint()
        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.train_loader_aug,self.test_loader, self.train_loader_for_CPs)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def freeze_backbone(self,is_first_session=False):
        # Freeze the parameters for ViT.
        if 'vit' in self.args['convnet_type']:
            if isinstance(self._network.convnet, nn.Module):
                for name, param in self._network.convnet.named_parameters():
                    if is_first_session:
                        if "head." not in name and "ssf_scale" not in name and "ssf_shift_" not in name: 
                            param.requires_grad = False
                    else:
                        param.requires_grad = False
        else:
            if isinstance(self._network.convnet, nn.Module):
                for name, param in self._network.convnet.named_parameters():
                    if is_first_session:
                        if "ssf_scale" not in name and "ssf_shift_" not in name: 
                            param.requires_grad = False
                    else:
                        param.requires_grad = False

    def show_num_params(self,verbose=False):
        # show total parameters and trainable parameters
        total_params = sum(p.numel() for p in self._network.parameters())
        logging.info(f'{total_params:,} total parameters.')
        total_trainable_params = sum(p.numel() for p in self._network.parameters() if p.requires_grad)
        logging.info(f'{total_trainable_params:,} training parameters.')
        if total_params != total_trainable_params and verbose:
            for name, param in self._network.named_parameters():
                if param.requires_grad:
                    print(name, param.numel())

    def _train(self, train_loader, train_loader_aug,test_loader, train_loader_for_CPs):
        self._network.to(self._device)
        if self._cur_task == 0 and self.args["model_name"] in ['ncm','joint_linear']:
             self.freeze_backbone()
        if self.args["model_name"] in ['joint_linear','joint_full']: 
            #this branch updates using SGD on all tasks and should be using classes and does not use a RP head
            if self.args["model_name"] =='joint_linear':
                assert self.args['body_lr']==0.0
            self.show_num_params()
            optimizer = optim.SGD([{'params':self._network.convnet.parameters()},{'params':self._network.fc.parameters(),'lr':self.args['head_lr']}], 
                                        momentum=0.9, lr=self.args['body_lr'],weight_decay=self.weight_decay)
            scheduler=optim.lr_scheduler.MultiStepLR(optimizer,milestones=[100000])
            logging.info("Starting joint training on all data using "+self.args["model_name"]+" method")
            self._init_train(train_loader, train_loader_aug,test_loader, optimizer, scheduler)
            self.show_num_params()
        else:
            #this branch is either CP updates only, or SGD on a PETL method first task only
            if self._cur_task == 0 and self.dil_init==False:
                if 'ssf' in self.args['convnet_type']:
                    self.freeze_backbone(is_first_session=True)
                if self.args["model_name"] != 'ncm':
                    #this will be a PETL method. Here, 'body_lr' means all parameters
                    self.show_num_params()
                    optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.args['body_lr'],weight_decay=self.weight_decay)
                    scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)
                    #train the PETL method for the first task:
                    logging.info("Starting PETL training on first task using "+self.args["model_name"]+" method")
                    self._init_train(train_loader, train_loader_aug,test_loader, optimizer, scheduler)
                    self.freeze_backbone()
                if self.args['use_RP'] and self.dil_init==False:
                    self.setup_RP()
            if self.is_dil and self.dil_init==False:
                self.dil_init=True
                self._network.fc.weight.data.fill_(0.0)
            self.replace_fc(train_loader_for_CPs)
            self.show_num_params()
        
    
    def setup_RP(self):
        self.initiated_G=False
        self._network.fc.use_RP=True
        if self.args['M']>0:
            #RP with M > 0
            M=self.args['M']
            # self._network.fc.weight = nn.Parameter(torch.Tensor(self._network.fc.out_features, M).to(device='cuda')) #num classes in task x M
            self._network.fc.weight = nn.Parameter(torch.Tensor(self._network.fc.out_features, M).to(self._device)) 
            self._network.fc.reset_parameters()
            self._network.fc.W_rand=torch.randn(self._network.fc.in_features,M).to(device='cuda')
            
            # self._network.fc.W_rand=torch.randn(self._network.fc.in_features,M).to(self._device)
            
            # self._network.fc.W_rand=  torch.cat((torch.eye(self._network.fc.in_features),torch.randn(self._network.fc.in_features,M - self._network.fc.in_features)), dim =1 ).to(device='cuda')
                
            self.W_rand=copy.deepcopy(self._network.fc.W_rand) #make a copy that gets passed each time the head is replaced
        else:
            #no RP, only decorrelation
            M=self._network.fc.in_features #this M is L in the paper
        self.Q=torch.zeros(M,self.total_classnum)
        self.G=torch.zeros(M,M)
    
    def classify_feats(self, prototypes, classes, feats, targets):
        # Classify new examples with prototypes and return classification error
        dist = torch.pow(prototypes[None, :] - feats[:, None], 2).sum(dim=2)  # Squared euclidean distance
        preds = F.log_softmax(-dist, dim=1)
        labels = (classes[None, :] == targets[:, None]).long().argmax(dim=-1)
        acc = (preds.argmax(dim=1) == labels).float().mean()
        return preds, labels, acc
    def supervised_contrastive_loss(self, f1, f2, labels, temperature=0.5):
        """
        Computes the Supervised Contrastive Loss using labels.
        
        Args:
            f1 (torch.Tensor): Features from the original images, shape (bs, d).
            f2 (torch.Tensor): Features from the augmented images, shape (bs, d).
            labels (torch.Tensor): Tensor of shape (bs,) containing labels for each feature.
            temperature (float): Temperature scaling factor. Default is 0.5.
        
        Returns:
            torch.Tensor: Computed supervised contrastive loss.
        """
        # Normalize features to unit vectors
        f1 = F.normalize(f1, dim=1)  # shape (bs, d)
        f2 = F.normalize(f2, dim=1)  # shape (bs, d)
        
        # Concatenate f1 and f2 to create a combined batch of features
        features = torch.cat([f1, f2], dim=0)  # shape (2 * bs, d)
        
        # Repeat labels for the augmented batch
        labels = torch.cat([labels, labels], dim=0)  # shape (2 * bs,)
        bs = f1.shape[0]
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T)  # shape (2 * bs, 2 * bs)
        
        # Scale similarities by temperature
        similarity_matrix /= temperature
        
        # Create a mask for positive pairs using labels
        label_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(f1.device)  # shape (2 * bs, 2 * bs)
        
        # Mask to exclude self-similarity (diagonal elements)
        self_mask = torch.eye(2 * bs, dtype=torch.bool).to(f1.device)
        label_mask[self_mask] = 0  # Remove self-similarities from positive mask
        
        # Compute the log-softmax of the similarity scores
        logits = similarity_matrix - torch.max(similarity_matrix, dim=1, keepdim=True)[0]  # Stability adjustment
        exp_logits = torch.exp(logits)
        
        # Masked normalization for the denominator (only negatives)
        denom = exp_logits.sum(dim=1, keepdim=True) - exp_logits * label_mask
        log_prob = logits - torch.log(denom + 1e-8)  # Avoid log(0) with epsilon
        
        # Compute the supervised contrastive loss
        loss = - (label_mask * log_prob).sum(dim=1) / label_mask.sum(dim=1)
        loss = loss.mean()
        
        return loss
    
    def w_function(self,x, k=10, x0=4):
    # Logistic sigmoid-based function
        return 1 - 1 / (1 + torch.exp(-k * (x - x0)))
    
    def _init_train(self, train_loader, train_loader_aug, test_loader, optimizer, scheduler):
        if self.args['meta']:
            logging.info("Training Started with self -supervised loss......")
        else:
            logging.info("Training Started with label CE loss......")
        weight_scale = self.w_function(torch.tensor(int(self.args['shot'])))
        logging.info(f"Weight scale value mse ...... {weight_scale}")
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            
            if self.args['meta']:
                
                losses = 0.0
                correct, total = 0, 0
                # labeled_iter = iter(train_loader_aug)
                
                for i, (_, inputs, inputs_aug, targets, targets_aug) in enumerate(train_loader):
                    # breakpoint()
                    # for i, (_, inputs_aug, targets_aug) in enumerate(train_loader_aug):
                    inputs, targets = inputs.to(self._device), targets.to(self._device)
                    inputs_aug, targets_aug = inputs_aug.to(self._device), targets_aug.to(self._device)
                    # features = self._network.convnet(torch.cat((inputs, inputs_aug),0))
                    outputs = self._network(torch.cat((inputs, inputs_aug),0))
                    features = outputs['features']
                    logits = outputs['logits']
                    f1, f2 = features.chunk(2)
                    # logits, labels, _ = self.classify_feats(f1,targets, f2, targets_aug)
                    # loss_ce = F.cross_entropy(logits[:inputs.shape[0]], targets)
                    # breakpoint()
                    loss_ce = F.cross_entropy(logits[:inputs.shape[0]], targets)
                    # loss_ce = F.cross_entropy(logits, torch.cat((targets,targets_aug)))
                    loss_mse = F.mse_loss(f1, f2)
                    # loss = loss_mse + loss_ce
                    loss = (weight_scale * loss_mse) + ((1 - weight_scale) * loss_ce)
                    # loss = loss_ce
                    # loss= loss_mse
                    # loss = self.supervised_contrastive_loss(f1, f2, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses += loss.item()
                         # total += len(targets)
                scheduler.step()
                # train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
                train_acc = 0
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['tuned_epoch'],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
                prog_bar.set_description(info)
                    
            else:
                losses = 0.0
                correct, total = 0, 0
                for i, (_, inputs, targets) in enumerate(train_loader):
                    # breakpoint()
                    inputs, targets = inputs.to(self._device), targets.to(self._device)
                    logits = self._network(inputs)["logits"]
                    loss = F.cross_entropy(logits, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses += loss.item()
                    _, preds = torch.max(logits, dim=1)
                    correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                    total += len(targets)
                scheduler.step()
                train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['tuned_epoch'],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
                prog_bar.set_description(info)

        logging.info(info)
        
    

   