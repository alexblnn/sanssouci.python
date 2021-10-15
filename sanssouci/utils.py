import numpy as np
from nilearn.input_data import NiftiMasker
from nilearn.datasets import get_data_dirs
from scipy import stats
import os
import json
from tqdm import tqdm
from joblib import Memory
from .lambda_calibration import get_permuted_p_values_one_sample
from .lambda_calibration import get_pivotal_stats
from .reference_families import linear_template


def get_data_driven_template_one_task(task, B=1000, smoothing_fwhm=4, seed=None):
    """
    Get data driven template for a single task (generally vs baseline)
    """

    data_path = get_data_dirs()[0]
    data_location = os.path.join(data_path, 'neurovault/collection_1952')
    paths = [data_location + '/' + path for path in os.listdir(data_location)]

    files_id = []

    for path in paths:
        if path.endswith(".json") and 'collection_metadata' not in path:
            f = open(path)
            data = json.load(f)
            files_id.append((data['relative_path'], data['file']))

    images_task1 = []
    for i in range(len(files_id)):
        if task in files_id[i][1]:
            img_path = files_id[i][0].split(sep='/')[1]
            images_task1.append(os.path.join(data_location, img_path))

    nifti_masker = NiftiMasker(smoothing_fwhm=smoothing_fwhm)

    fmri_input = nifti_masker.fit_transform(images_task1)

    pval0 = get_permuted_p_values_one_sample(fmri_input, B=B, seed=seed)

    pval0_quantiles = np.sort(pval0, axis=0)
    return pval0_quantiles


def get_data_driven_template_two_tasks(task1, task2, B=100, seed=None):
    """
    Get data-driven template for task1 vs task2
    """

    data_path = get_data_dirs()[0]
    data_location = os.path.join(data_path, 'neurovault/collection_1952')
    paths = [data_location + '/' + path for path in os.listdir(data_location)]

    files_id = []

    for path in paths:
        if path.endswith(".json") and 'collection_metadata' not in path:
            f = open(path)
            data = json.load(f)
            files_id.append((data['relative_path'], data['file']))

    subjects1, subjects2 = [], []

    images_task1 = []
    for i in range(len(files_id)):
        if task1 in files_id[i][1]:
            img_path = files_id[i][0].split(sep='/')[1]
            images_task1.append(os.path.join(data_location, img_path))
            filename = files_id[i][1].split(sep='/')[6]
            subjects1.append(filename.split(sep='base')[1])

    images_task1 = np.array(images_task1)

    images_task2 = []
    for i in range(len(files_id)):
        if task2 in files_id[i][1]:
            img_path = files_id[i][0].split(sep='/')[1]
            images_task2.append(os.path.join(data_location, img_path))
            filename = files_id[i][1].split(sep='/')[6]
            subjects2.append(filename.split(sep='base')[1])

    images_task2 = np.array(images_task2)

    common_subj = list(set(subjects1) & set(subjects1))
    indices1 = [subjects1.index(common_subj[i]) for i in range(len(common_subj))]
    indices2 = [subjects2.index(common_subj[i]) for i in range(len(common_subj))]

    # Let's process the data

    nifti_masker = NiftiMasker(smoothing_fwhm=4)
    nifti_masker.fit(np.concatenate([images_task1[indices1], images_task2[indices2]]))

    fmri_input1 = nifti_masker.transform(images_task1[indices1])
    fmri_input2 = nifti_masker.transform(images_task2[indices2])

    fmri_input = fmri_input1 - fmri_input2
    stats_, p_values = stats.ttest_1samp(fmri_input, 0)
    # add underscore to stats to avoid confusion with stats package

    pval0 = get_permuted_p_values_one_sample(fmri_input, B=B, seed=seed)
    pval0_quantiles = np.sort(pval0, axis=0)
    return pval0_quantiles


def get_processed_input(task1, task2):
    """
    Get processed input for Neurovault tasks
    """
    data_path = get_data_dirs()[0]
    data_location = os.path.join(data_path, 'neurovault/collection_1952')
    paths = [data_location + '/' + path for path in os.listdir(data_location)]

    files_id = []

    for path in paths:
        if path.endswith(".json") and 'collection_metadata' not in path:
            f = open(path)
            data = json.load(f)
            files_id.append((data['relative_path'], data['file']))

    subjects1, subjects2 = [], []

    images_task1 = []
    for i in range(len(files_id)):
        if task1 in files_id[i][1]:
            img_path = files_id[i][0].split(sep='/')[1]
            images_task1.append(os.path.join(data_location, img_path))
            filename = files_id[i][1].split(sep='/')[6]
            subjects1.append(filename.split(sep='base')[1])

    images_task1 = np.array(images_task1)

    images_task2 = []
    for i in range(len(files_id)):
        if task2 in files_id[i][1]:
            img_path = files_id[i][0].split(sep='/')[1]
            images_task2.append(os.path.join(data_location, img_path))
            filename = files_id[i][1].split(sep='/')[6]
            subjects2.append(filename.split(sep='base')[1])

    images_task2 = np.array(images_task2)

    common_subj = list(set(subjects1) & set(subjects1))
    indices1 = [subjects1.index(common_subj[i]) for i in range(len(common_subj))]
    indices2 = [subjects2.index(common_subj[i]) for i in range(len(common_subj))]

    # Let's process the data

    nifti_masker = NiftiMasker(smoothing_fwhm=4)
    nifti_masker.fit(np.concatenate([images_task1[indices1], images_task2[indices2]]))
    fmri_input1 = nifti_masker.transform(images_task1[indices1])
    fmri_input2 = nifti_masker.transform(images_task2[indices2])

    fmri_input = fmri_input1 - fmri_input2

    return fmri_input, nifti_masker


def calibrate_simes(fmri_input, alpha, k_min, k_max, B=100, seed=None):
    p = fmri_input.shape[1]
    pval0 = get_permuted_p_values_one_sample(fmri_input, B=B, seed=seed)
    piv_stat = get_pivotal_stats(pval0, K=k_max)
    lambda_quant = np.quantile(piv_stat, alpha)
    simes_thr = linear_template(lambda_quant, k_max, p)

    return pval0, simes_thr
