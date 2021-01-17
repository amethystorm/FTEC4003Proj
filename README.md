# FTEC4003 Course Project

Group ID: 4

Sida Huang$\qquad$ $\qquad$1155124414$\qquad$ $\qquad$sdhuang@link.cuhk.edu.hk

Xiran Zhang$\qquad$ $\qquad$1155124428$\qquad$ $\qquad$1155124428@link.cuhk.edu.hk



## File description

- README.md: This introduction file
- FTEC4003_report_04.pdf: Project report (Page 2 - 5: Task 1 report; Page 6 - 13: Task 2 report)
- submission_1_DT.csv: Task 1 submission, Decision Tree, f1_score = 0.29789929272669696
- submission_1_GNB.csv: Task 1 submission, Gaussian Naive Bayes, f1_score = 0.3895608729950039
- submission_2.csv: Task 2 submission, f1_score = 0.6351209253417456

- Task-1_submission: Code folder for task 1. Please see the detailed introduction in our project report.
  - insurance-train.csv: Training data
  - insurance-test.csv: Testing data
  - task1-DT.py: Decision Tree
  - task1-KNN.py: K Nearest Neighbors
  - task1-Bayes.py: Naive Bayes
  - task1-SVM.py: Support Vector Machine
  - task1-Bagging.py: Bagging
- Task-2_submission: Code folder for task 2. Please see the detailed introduction in our project report.
  - data: data after one-hot encoding
    - onehot_train.csv: Training data
    - onehot_test.csv: Testing data
  - default_models.py: 16 default models
  - class_weight.py: tuning class_weight parameter
  - imbalanced.py: resampling using imbalanced-learn
  - smotevariants.py: over-sampling using smote_variants
  - final_model.py: generating submission_2.csv



This is our submission for the second time. We just modified the result of SVM in task 1 compared with the first submission.