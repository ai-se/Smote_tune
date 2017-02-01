## Fault Prediction -- pre-release name

The following are steps to install.

  - Install virtual environment. (Optional but recommended)
  - Then run the following:
```sh
  $ cd fault_prediction # Basically set cwd to fault_prediction
  $ pip install -e .
  $ python -m tests.test _test filename
```
  - filename just "ant"
  - Magic

## Experiments:
- 6 Learners [KNN, NB, SVM, LR, DT, RF]
- with smote and without smote, 5x5 cross val
- only 14 datasets, other datasets from openscience were very small.
- compared against 4 measures, [Precision, recall, accuracy, f_score]

## Conclusions:
- Considering all the measure almost all learners performed the best. But in most cases we saw, Random forest to win.
- Smote is needed for highly imbalanced classes.
- Runtimes are within 5-20 mins for each dataset with 6 learners repeating 25 times. Only 2 datasets took about 2hours each.

## Results: 
# With Smote:

![file](https://github.com/amritbhanu/fss16591/raw/master/project/Accuracy_smote.png)

![file](https://github.com/amritbhanu/fss16591/raw/master/project/Precision_smote.png)

![file](https://github.com/amritbhanu/fss16591/raw/master/project/Recall_smote.png)

![file](https://github.com/amritbhanu/fss16591/raw/master/project/F_score_smote.png)


# Without Smote:

![file](https://github.com/amritbhanu/fss16591/raw/master/project/Accuracy_nosmote.png)

![file](https://github.com/amritbhanu/fss16591/raw/master/project/Precision_nosmote.png)

![file](https://github.com/amritbhanu/fss16591/raw/master/project/Recall_nosmote.png)

![file](https://github.com/amritbhanu/fss16591/raw/master/project/F_score_nosmote.png)
