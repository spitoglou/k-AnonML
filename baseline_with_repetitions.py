#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
import csv
import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
import typer
import click
import logging
from rich.console import Console
from rich.table import Table
from rich import box
from rich.logging import RichHandler
import utils.utility as util
from basic_mondrian.anonymizer import get_result_one
from basic_mondrian.utils.read_adult_data import read_tree
from clustering_based.anonymizer import get_result_one as cb_get_result_one
from elemam.main import main as emain
from generalization.generalization import age, hierarchy, l1sub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from top_down_greedy.anonymizer import tdg_get_result_one
from utils.data import read_raw, write_anon
from utils.types import AnonMethod, Classifier, Dataset, MLRes


def setup_logging(verbose: int, debug: bool):
    """Setup logging with appropriate level based on verbosity"""
    if debug:
        level = logging.DEBUG
    elif verbose >= 2:
        level = logging.DEBUG
    elif verbose >= 1:
        level = logging.INFO
    else:
        level = logging.WARNING
        
    # Create our logger
    logger = logging.getLogger("k-anonymity")
    logger.setLevel(level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Add Rich handler
    handler = RichHandler(rich_tracebacks=True)
    handler.setLevel(level)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Prevent propagation to root logger to avoid matplotlib spam
    logger.propagate = False
    
    return logger


def display_results_table(results_data: list, baseline: float, dataset: str, anon_method: str, classifier: str):
    """Display experiment results in a Rich table"""
    console = Console()
    
    # Create table
    table = Table(title=f"k-Anonymity Experiment Results: {dataset.upper()} dataset with {anon_method.upper()} algorithm and {classifier.upper()} classifier", box=box.ROUNDED)
    
    # Add columns
    table.add_column("k", justify="center", style="cyan", no_wrap=True)
    if any(row['s'] != 0 for row in results_data):  # Only show s column if OLA
        table.add_column("s", justify="center", style="magenta", no_wrap=True)
    table.add_column("Accuracy (%)", justify="right", style="green")
    table.add_column("Precision", justify="right", style="yellow")
    table.add_column("Recall", justify="right", style="blue")
    table.add_column("F1-Score", justify="right", style="red")
    table.add_column("Accuracy Drop", justify="right", style="bright_red")
    
    # Add baseline row
    baseline_row = ["Baseline", f"{baseline:.2f}", "-", "-", "-", "0.00"]
    if any(row['s'] != 0 for row in results_data):
        baseline_row.insert(1, "-")  # Add s column value for baseline
    table.add_row(*baseline_row, style="bold")
    
    # Add separator
    table.add_section()
    
    # Add results rows
    for result in results_data:
        accuracy = result['accuracy']
        accuracy_drop = baseline - accuracy
        
        row_data = [
            str(result['k']),
            f"{accuracy:.2f}",
            f"{result['precision']:.3f}",
            f"{result['recall']:.3f}",
            f"{result['f1_score']:.3f}",
            f"{accuracy_drop:.2f}"
        ]
        
        # Add s column if OLA
        if any(row['s'] != 0 for row in results_data):
            row_data.insert(1, str(result['s']))
            
        # Color code based on accuracy drop
        if accuracy_drop <= 2:
            style = "green"
        elif accuracy_drop <= 5:
            style = "yellow"  
        else:
            style = "red"
            
        table.add_row(*row_data, style=style)
    
    console.print(table)
    console.print(f"\n[bold]Summary:[/bold] Tested k-anonymity from {results_data[0]['k']} to {results_data[-1]['k']} with {anon_method.upper()} algorithm")
    console.print(f"[bold]Best Result:[/bold] k={min(results_data, key=lambda x: baseline - x['accuracy'])['k']} (lowest accuracy drop: {min(baseline - r['accuracy'] for r in results_data):.2f}%)")

def run_experiment(dataset: str, anon_method: str, classifier: str, start_k: int = 2, stop_k: int = 100, step_k: int = 1, start_s: int = 3, stop_s: int = 3, step_s: int = 1, metric: str = 'gweight', cb_alg: str = 'knn', debug: bool = False, verbose: int = 0):
    # Setup logging
    logger = setup_logging(verbose, debug)
    
    sns.set()  # set defaults
    rnd = 42
    np.random.seed(rnd)

    # Log experiment configuration
    logger.info(f"[EXPERIMENT] Starting k-anonymity experiment")
    logger.info(f"[DATASET] {dataset.upper()}")
    logger.info(f"[CLASSIFIER] {classifier.upper()}")
    logger.info(f"[ANONYMIZATION] {anon_method.upper()}")
    logger.info(f"[K-RANGE] {start_k} to {stop_k} (step {step_k})")
    
    if anon_method == 'ola':
        logger.info(f"[SUPPRESSION] {start_s} to {stop_s} (step {step_s})")
        logger.info(f"[METRIC] {metric}")
    elif anon_method == 'cb':
        logger.info(f"[CLUSTERING] {cb_alg}")

    # Global Parameter
    k_range = range(start_k, stop_k + 1, step_k)
    total_k_values = len(list(k_range))

    # OLA Parameter
    s_range = [0]
    if anon_method == 'ola':
        # Suppression OLA
        s_range = range(start_s, stop_s + 1, step_s)
        
    total_s_values = len(list(s_range))
    total_experiments = total_k_values * total_s_values
    logger.info(f"[TOTAL] {total_experiments} experiments to run")
        
    # Initialize results collection
    all_results = []

    # define necessary paths

    # Data path
    path = os.path.join('datasets', dataset, '')  # trailing /
    # Dataset path
    data_path = os.path.join(path, f'{dataset}.csv')
    # Generalization hierarchies path
    gen_path = os.path.join('generalization', 'hierarchies', dataset, '')  # trailing /
    # folder for all results
    res_folder = os.path.join('results', dataset, anon_method, datetime.now().isoformat().replace(':', '_'))

    # ML results path
    output_path = os.path.join(res_folder, f'{dataset}_{os.path.basename(anon_method)}_{os.path.basename(classifier)}_k_{stop_k}.csv')
    # path for anonymized datasets
    anon_folder = os.path.join(res_folder, 'anon_dataset', '')  # trailing /
    # path for pickled numeric values
    numeric_folder = os.path.join(res_folder, 'numeric')
    # save ML features
    features_file = os.path.join(res_folder, 'features.csv')

    # create path needed for results recursively
    os.makedirs(anon_folder)
    os.makedirs(numeric_folder)

    xgb_eval_metric = 'error'

    # reading in the data
    logger.info(f"[LOADING] Dataset from: {data_path}")
    data = pd.read_csv(data_path, delimiter=';')
    logger.info(f"[LOADED] {data.shape[0]} entries, {data.shape[1]} attributes")
    print('Original Data: ' + str(data.shape[0]) + ' entries, ' + str(data.shape[1]) + ' attributes')

    ATT_NAMES = list(data.columns)

    if dataset == Dataset.CMC:
        QI_INDEX = [1, 2, 4]
        target_var = 'method'
        IS_CAT2 = [False, True, False]
        max_numeric = {"age": 32.5, "children": 8}
        xgb_eval_metric = 'merror'

    elif dataset == Dataset.MGM:
        QI_INDEX = [1, 2, 3, 4, 5]
        target_var = 'severity'
        IS_CAT2 = [True, False, True, True, True]
        max_numeric = {"age": 50.5}

    elif dataset == Dataset.CAHOUSING:
        QI_INDEX = [1, 2, 3, 8, 9]
        target_var = 'ocean_proximity'
        IS_CAT2 = [False, False, False, False, False]
        max_numeric = {"latitude": 119.33, "longitude": 37.245, "housing_median_age": 32.5,
                       "median_house_value": 257500, "median_income": 5.2035}
        xgb_eval_metric = 'merror'

    elif dataset == Dataset.ADULT:
        QI_INDEX = [1, 2, 3, 4, 5, 6, 7, 8]
        target_var = 'salary-class'
        IS_CAT2 = [True, False, True, True, True, True, True, True]
        max_numeric = {"age": 50.5}

    QI_NAMES = list(np.array(ATT_NAMES)[QI_INDEX])
    IS_CAT = [True] * len(QI_INDEX)
    SA_INDEX = [index for index in range(len(ATT_NAMES)) if index not in QI_INDEX]
    SA_var = [ATT_NAMES[i] for i in SA_INDEX]

    # one hot encoding for all categorical values
    one_hot_original = [col for i, col in enumerate(data[QI_NAMES].columns) if IS_CAT2[i]]
    one_hot_anon = one_hot_original

    if anon_method == AnonMethod.OLA:
        gen_strat = [hierarchy(gen_path + dataset, elem) for elem in QI_NAMES]
        # How often a QI can be generalized
        max_gen_level = [len(elem[1]) for elem in gen_strat]

    # override auto parameters as needed
    if dataset == Dataset.ADULT:
        max_gen_level = [1, 4, 1, 2, 3, 2, 2, 2]
        gen_strat = [
            l1sub, age, l1sub,
            hierarchy(gen_path + dataset, 'marital-status'),
            hierarchy(gen_path + dataset, 'education'),
            hierarchy(gen_path + dataset, 'native-country'),
            hierarchy(gen_path + dataset, 'workclass'),
            hierarchy(gen_path + dataset, 'occupation')
        ]

    elif dataset == Dataset.CAHOUSING:
        SA_var = ['ID', 'ocean_proximity']

    elif dataset == Dataset.CMC:
        SA_var = ['ID', 'method']

    # Experiments on original Data

    # label encoding of the target variable
    data[target_var] = data[target_var].astype('category').cat.codes

    # one hot encoding of categorical variables needed for the classification task
    data2 = pd.get_dummies(data, columns=one_hot_original, drop_first=True)

    # creating the ground truth (target variable) vector and removing target variable and ID from the dataset
    y = data[target_var]
    X = data2.drop(SA_var, axis=1)

    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    # split the dataset into training and testing set
    logger.info(f"[SPLIT] 70% train, 30% test")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rnd)
    logger.info(f"[SAMPLES] Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # creating a classifer
    logger.info(f"[CLASSIFIER] Creating {classifier.upper()}")
    clf = util.create_classifier(classifier, dataset)
    if verbose > 0 or debug:
        print(clf)
    else:
        logger.debug(f"[CLASSIFIER] {clf.__class__.__name__} configured")

    # train the model using the training sets
    logger.info(f"[TRAINING] Classifier on original data...")
    if classifier == Classifier.XGB:
        clf.fit(X_train, y_train, eval_metric=[xgb_eval_metric], eval_set=[
                (X_train, y_train), (X_test, y_test)], early_stopping_rounds=40)
    else:
        clf.fit(X_train, y_train)

    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)
    if verbose > 0:
        util.show_classifier_metrics(y_train, pred_train, y_test, pred_test)

    # calculating the original data baseline accuracy
    baseline_metrics = util.get_classifier_metrics(np.asarray(y_test), pred_test)
    baseline = baseline_metrics[0]  # accuracy
    zero_rule_baseline = util.zero_rule_baseline(y_test)
    logger.info(f"[BASELINE] Accuracy: {baseline:.2f}%")
    print('Original data baseline: %f%%' % (baseline))
    print('Zero-Rule baseline: %f%%' % (zero_rule_baseline))

    # Data Anonymization and repeated experiments (with different k)
    logger.info(f"[STARTING] Anonymization experiments...")
    experiment_count = 0

    with open(features_file, 'a+') as f_file:
        writer = csv.writer(f_file)
        nodes_count = 1
        raw_data, header = read_raw(path, numeric_folder, dataset, QI_INDEX, IS_CAT)
        ATT_TREES = read_tree(gen_path, numeric_folder, dataset, ATT_NAMES, QI_INDEX, IS_CAT)
        for s_idx, s in enumerate(s_range):
            logger.info(f"[SUPPRESSION] s={s} ({s_idx + 1}/{total_s_values})")
            s_folder = os.path.join(anon_folder, 's_' + str(s))
            os.mkdir(s_folder)
            ml_res = MLRes()
            for k_idx, k in enumerate(k_range):
                experiment_count += 1
                logger.info(f"[ANONYMIZING] k={k} ({k_idx + 1}/{total_k_values}) - Experiment {experiment_count}/{total_experiments}")
                
                import time
                start_time = time.time()
                anon_data = None
                if anon_method == AnonMethod.MONDRIAN:
                    anon_data = get_result_one(ATT_TREES, raw_data, k, path, QI_INDEX, SA_INDEX, logger)
                elif anon_method == AnonMethod.TDG:
                    anon_data = tdg_get_result_one(ATT_TREES, raw_data, k, path, QI_INDEX, SA_INDEX, logger)
                elif anon_method == AnonMethod.CB:
                    anon_data = cb_get_result_one(ATT_TREES, raw_data, k, path, QI_INDEX, SA_INDEX, cb_alg, logger)
                elif anon_method == AnonMethod.OLA:
                    logger.debug(f"[OLA] Running with metric={metric}...")
                    # Anonymize data with OLA
                    anon_data, gen_level_array = emain(raw_data, k, gen_strat, max_gen_level,
                                                       QI_INDEX, metric, res_folder, suppression_rate=s)

                # Timing is now handled by the algorithm loggers
                
                # Write anonymized data in csv file
                nodes_count = write_anon(s_folder, anon_data, header, k, s, dataset)

                for node in range(nodes_count):
                    logger.debug(f"[EVALUATING] Node {node + 1}/{nodes_count}...")
                    # reading in the anonymized data
                    anon_data = pd.read_csv(os.path.join(
                        s_folder, dataset + "_anonymized_" + str(k) + '_' + str(node) + ".csv"), delimiter=';')
                    anon_info = f'K: {k} S: {s} Node: {node} | Anonymized Data: {anon_data.shape[0]} entries, {anon_data.shape[1]} attributes'
                    print(anon_info)
                    logger.debug(f"[DATA] {anon_info}")
                    # we have to sort the data with respect to ID (in case a anonymization algorithm rearranges the entries)
                    anon_data = anon_data.sort_values(by=['ID'])

                    # label encoding of the target variable
                    anon_data[target_var] = anon_data[target_var].astype('category').cat.codes

                    # creating the ground truth (target variable) vector and removing target variable and ID from the dataset
                    y = anon_data[target_var]
                    X = anon_data.drop(SA_var, axis=1)

                    for index_row, row in X.iterrows():
                        cat_iter = iter(IS_CAT2)
                        for index_col, col in row.items():
                            # only quasi identifiers
                            if index_col not in QI_NAMES:
                                continue

                            # only non categorical attributes
                            if next(cat_iter):
                                continue

                            # replace suppressed value with highest value of according attribute
                            if col == '*':
                                newval = max_numeric.get(index_col)
                                if newval is None:
                                    logger.warning(f"[ERROR] {max_numeric.get(index_col)} index: {index_col}")
                                X.at[index_row, index_col] = newval
                                continue

                            try:
                                # check if value is a range e.g. a-c
                                val = col.split('-')
                                if len(val) == 1:
                                    continue
                                if val[0] == "" or val[1] == "":
                                    continue

                                # replace range value with mean
                                newval = (float(val[0]) + float(val[1])) / 2
                                if newval is None:
                                    logger.warning(f"[ERROR] {max_numeric.get(index_col)} index: {index_col}")
                                X.at[index_row, index_col] = newval
                            except AttributeError:
                                pass

                    # replace all categorical value with numeric values
                    for qi in max_numeric.keys():
                        logger.debug(f"[PROCESSING] QI column: {qi}")
                        X[qi] = pd.to_numeric(X[qi])

                    # one hot encoding of categorical variables needed for the classification task
                    X = pd.get_dummies(X, columns=one_hot_anon, drop_first=True)

                    logger.debug(f"[DATAFRAME] Shape: {X.shape}")

                    scaler = MinMaxScaler()
                    scaler.fit(X)
                    X = scaler.transform(X)

                    # write feature
                    writer.writerow(X.shape)

                    # spliting the dataset into training and testing set
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rnd)

                    # creating a classifer
                    clf = util.create_classifier(classifier, dataset)

                    # train the model using the training sets
                    if classifier == Classifier.XGB:
                        clf.fit(
                            X_train, y_train, eval_metric=[xgb_eval_metric], eval_set=[
                                (X_train, y_train), (X_test, y_test)], early_stopping_rounds=40
                        )

                    clf.fit(X_train, y_train)
                    pred_train = clf.predict(X_train)
                    pred_test = clf.predict(X_test)

                    # append accuracty, precision, recall and f1 score to the ML results
                    current_metrics = util.get_classifier_metrics(np.asarray(y_test), pred_test)
                    for i, res in enumerate(current_metrics):
                        ml_res[i].append(res)

                    # Collect results for final table
                    all_results.append({
                        'k': k,
                        's': s,
                        'accuracy': current_metrics[0],
                        'precision': current_metrics[1],
                        'recall': current_metrics[2],
                        'f1_score': current_metrics[3]
                    })

                    logger.info(f"[RESULT] k={k}, s={s}: Accuracy={current_metrics[0]:.2f}%, F1={current_metrics[3]:.3f}, Drop={baseline - current_metrics[0]:.2f}%")
                    # Verbose output is handled by logger.info above

                # For OLA debugging
                if debug:
                    util.write_results(s, ml_res, anon_method, output_path, num=i)
                    ml_res = MLRes()

            util.write_results(s, ml_res, anon_method, output_path)

    if verbose > 1:
        logger.debug(f"[ML_RESULTS] {ml_res}")
            
    # Display final results table
    if all_results:
        logger.info(f"[COMPLETED] Processed {len(all_results)} configurations")
        best_result = min(all_results, key=lambda x: baseline - x['accuracy'])
        logger.info(f"[BEST] k={best_result['k']}, s={best_result['s']}, accuracy_drop={baseline - best_result['accuracy']:.2f}%")
        display_results_table(all_results, baseline, dataset, anon_method, classifier)
    else:
        logger.warning("[WARNING] No results to display")


app = typer.Typer(help="Anonymize data utilising different algorithms and analyse the effects of the anonymization on the data")

def validate_k_params(start_k: int, stop_k: int, step_k: int):
    """Validate k-anonymity parameters"""
    if start_k < 2:
        raise typer.BadParameter("invalid start_k value")
    if stop_k < start_k:
        raise typer.BadParameter("stop_k needs to be greater than start_k")
    if step_k < 1:
        raise typer.BadParameter("invalid step_k value")
    if start_k != stop_k and start_k + step_k > stop_k:
        raise typer.BadParameter("invalid step_k value")

@app.command()
def mondrian(
    dataset: Dataset = typer.Argument(help="The dataset used for anonymization"),
    classifier: Classifier = typer.Argument(help="Machine learning classifier"),
    start_k: int = typer.Option(2, "--start-k", help="Initial value for k of k-anonymity"),
    stop_k: int = typer.Option(100, "--stop-k", help="Last value for k of k-anonymity"),
    step_k: int = typer.Option(1, "--step-k", help="Step for increasing k of k-anonymity"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debugging"),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True, help="Verbose output")
):
    """Run Mondrian anonymization algorithm"""
    validate_k_params(start_k, stop_k, step_k)
    run_experiment(str(dataset), 'mondrian', str(classifier), start_k, stop_k, step_k, debug=debug, verbose=verbose)

@app.command()
def ola(
    dataset: Dataset = typer.Argument(help="The dataset used for anonymization"),
    classifier: Classifier = typer.Argument(help="Machine learning classifier"),
    start_k: int = typer.Option(2, "--start-k", help="Initial value for k of k-anonymity"),
    stop_k: int = typer.Option(100, "--stop-k", help="Last value for k of k-anonymity"),
    step_k: int = typer.Option(1, "--step-k", help="Step for increasing k of k-anonymity"),
    start_s: int = typer.Option(3, "--start-s", help="Initial value for suppression of ola"),
    stop_s: int = typer.Option(3, "--stop-s", help="Last value for suppression of ola"),
    step_s: int = typer.Option(1, "--step-s", help="Step for increasing suppression of ola"),
    metric: str = typer.Option('gweight', "--metric", "-m", help="OLA metric", 
                              click_type=click.Choice(['none', 'gweight', 'prec', 'aecs', 'dm', 'ent'])),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debugging"),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True, help="Verbose output")
):
    """Run OLA anonymization algorithm"""
    validate_k_params(start_k, stop_k, step_k)
    run_experiment(str(dataset), 'ola', str(classifier), start_k, stop_k, step_k, 
                   start_s, stop_s, step_s, metric, debug=debug, verbose=verbose)

@app.command()
def tdg(
    dataset: Dataset = typer.Argument(help="The dataset used for anonymization"),
    classifier: Classifier = typer.Argument(help="Machine learning classifier"),
    start_k: int = typer.Option(2, "--start-k", help="Initial value for k of k-anonymity"),
    stop_k: int = typer.Option(100, "--stop-k", help="Last value for k of k-anonymity"),
    step_k: int = typer.Option(1, "--step-k", help="Step for increasing k of k-anonymity"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debugging"),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True, help="Verbose output")
):
    """Run Top-Down Greedy anonymization algorithm"""
    validate_k_params(start_k, stop_k, step_k)
    run_experiment(str(dataset), 'tdg', str(classifier), start_k, stop_k, step_k, debug=debug, verbose=verbose)

@app.command()
def cb(
    dataset: Dataset = typer.Argument(help="The dataset used for anonymization"),
    classifier: Classifier = typer.Argument(help="Machine learning classifier"),
    start_k: int = typer.Option(2, "--start-k", help="Initial value for k of k-anonymity"),
    stop_k: int = typer.Option(100, "--stop-k", help="Last value for k of k-anonymity"),
    step_k: int = typer.Option(1, "--step-k", help="Step for increasing k of k-anonymity"),
    cb_alg: str = typer.Option('knn', "--cb-alg", help="Algorithm for cluster based anonymization",
                              click_type=click.Choice(['knn', 'kmember', 'oka'])),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debugging"),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True, help="Verbose output")
):
    """Run Clustering-Based anonymization algorithm"""
    validate_k_params(start_k, stop_k, step_k)
    run_experiment(str(dataset), 'cb', str(classifier), start_k, stop_k, step_k, cb_alg=cb_alg, debug=debug, verbose=verbose)

if __name__ == "__main__":
    app()
