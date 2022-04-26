import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class Feature_Selector:
    def __init__(self, metric, num_features):
        self.metric = metric[0]
        self.metric_callback = getattr(self, self.metric)
        self.print_callback  = getattr(self, "print_" + self.metric + "_results")
        self.optimal_condition = getattr(np, metric[1])
        self.num_features = num_features
        self.feature_order = [] # This is the feature order by selection
        self.feature_order_id = [] # This is purely used in the display function -- 

    def run_selection(self, data):
        features = data["feature_list"]
        features_included = []
        features_remaining = features
        for fi in range(0, self.num_features):
            criterion = self.metric_callback(data, features_included, features_remaining)
            chosen_feature = self.optimal_condition(criterion)
            self.feature_order.append(features_remaining[chosen_feature])
            self.feature_order_id.append(chosen_feature)
            features_included = self.feature_order
            features_remaining.remove(features_remaining[chosen_feature])

    def accuracy(self, data, features_included, features_remaining):

        if not hasattr(self, "mean_accuracy"):
            self.feature_list = data["feature_list"]
            self.mean_accuracy = {}
            self.std_accuracy = {}

        num_subjects = len(np.unique(data["subject"]))
        num_reps     = len(np.unique(data["rep"]))
        
        mean_accuracy = np.zeros(len(features_remaining))
        std_accuracy  = np.zeros(len(features_remaining))

        prior_features = np.array([])
        for feature in features_included:
            prior_features = np.hstack([prior_features, data[feature]]) if prior_features.size else data[feature]
        
        for i,feature in enumerate(features_remaining):
            iteration_features = np.hstack([prior_features, data[feature]])  if prior_features.size else data[feature]
            feature_accuracy = np.zeros((num_subjects, num_reps))
            # do a k-fold cross-validation within subject
            for subject in range(num_subjects):
                subject_class    = data["class"][data["subject"] == subject]
                subject_rep      = data["rep"]  [data["subject"] == subject]
                subject_features = iteration_features[data["subject"] == subject,:]

                for rep in range(num_reps):
                    train_class    = subject_class[subject_rep != rep]
                    train_features = subject_features[subject_rep != rep, :]
                    test_class     = subject_class[subject_rep == rep]
                    test_features  = subject_features[subject_rep == rep, :]

                    mdl = LinearDiscriminantAnalysis()
                    mdl.fit(train_features, train_class)
                    predictions = mdl.predict(test_features)
                    feature_accuracy[subject, rep] = sum(predictions == test_class) / test_class.shape[0]
            mean_accuracy[i] = np.mean(np.mean(feature_accuracy, axis=1))
            std_accuracy[i]  = np.std(np.mean(feature_accuracy,axis=1))

        self.mean_accuracy[len(self.mean_accuracy.keys())] = mean_accuracy
        self.std_accuracy[len(self.std_accuracy.keys())] = std_accuracy

        return mean_accuracy

    def print_results(self):
        self.print_callback()
    
    def print_accuracy_results(self):
        # longest feature name
        longest = 11
        for f in self.feature_order:
            if longest < len(f):
                longest = len(f)
        header_row = "iter".center(longest)
        for f in range(len(self.feature_order)):
            header_row += "|" + self.feature_order[f].center(longest)
        print(header_row)
        print('='*((longest+1)*(len(self.feature_order)+1)))


        for i in range(len(self.feature_order)):
            row = str(i).center(longest)
            mean_acc = self.mean_accuracy[i][self.feature_order_id[i:]] * 100
            std_acc  = self.std_accuracy[i][self.feature_order_id[i:]] * 100

            for j in range(i):
                row += "|" + " ".center(longest)
            for ii,f in enumerate(range(i, len(self.feature_order))):
                row += "|" + ("{:.1f}+{:.1f}".format(mean_acc[ii],std_acc[ii])).center(longest)
            print(row)
