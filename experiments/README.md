All experiments are saved in the `./experiments` folder and saved separately according to the name.   
The directory structure of each experiment is as follows:
``` 
├── experiments    		       # Experiment results folder
│   ├── Example          # An experiment
│        ├── args.pkl			    # Saved configuration file (pkl format)
│        ├── args.txt			    # Saved configuration file (txt format)
│        ├── best_model.pth 		# Best performance model saved
│        ├── events.out.tfevents.00.Server	# Tensorboard log files (including loss, acc and auc, etc.)
│        ├── latest_model.pth		# Latest model saved
│        ├── log_2021-01-14-23-09.csv 	# csv log files (including val loss, acc and auc, etc.)
│        ├── performances.txt		# Performance on the testset
│        ├── Precision_recall.png  	# P-R curve on the testset
│        ├── result_img		        # Visualized results of the testset
│        ├── result.npy		        # Pixel probability prediction result
│        ├── ROC.png			    # ROC curve on the testset
│        ├── sample_input_imgs.png	# Input image patches example
│        ├── sample_input_masks.png	# Input label example
│        ├── test_log.txt		    # Training process log
│        └── train_log.txt		    # Test process log
```
PS: The `./experiments/Example` is just an example, which `best_model.pth`、`latest_model.pth` and `result.npy` are empty files and cannot be used directly. You need to retrain the model.